#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#define NUM_VECTORS 8192
#define VECTOR_ELEMS 256
#define K 200
#define TILE_SIZE 32

using embedding = std::array<float, VECTOR_ELEMS>;
using topk_elem = std::tuple<float, int, int>;

struct TopKHeap {
  std::array<topk_elem, K> data{};
  int size = 0;

  bool full() const { return size == K; }

  float threshold() const {
    return full() ? std::get<0>(data[0])
                  : -std::numeric_limits<float>::max();
  }

  float best_score() const {
    return std::get<0>(*std::max_element(
        data.begin(), data.begin() + size,
        [](const topk_elem &a, const topk_elem &b) {
          return std::get<0>(a) < std::get<0>(b);
        }));
  }

  void push(float score, int first, int second) {
    if (size < K) {
      data[size] = {score, first, second};
      heapify_up(size);
      ++size;
    } else if (score > std::get<0>(data[0])) {
      data[0] = {score, first, second};
      heapify_down(0);
    }
  }

  void heapify_up(int idx) {
    while (idx > 0) {
      const int parent = (idx - 1) / 2;
      if (std::get<0>(data[parent]) <= std::get<0>(data[idx]))
        break;
      std::swap(data[parent], data[idx]);
      idx = parent;
    }
  }

  void heapify_down(int idx) {
    while (idx < size) {
      const int left = idx * 2 + 1;
      const int right = idx * 2 + 2;
      int smallest = idx;
      if (left < size && std::get<0>(data[left]) < std::get<0>(data[smallest]))
        smallest = left;
      if (right < size &&
          std::get<0>(data[right]) < std::get<0>(data[smallest]))
        smallest = right;
      if (idx == smallest)
        break;
      std::swap(data[smallest], data[idx]);
      idx = smallest;
    }
  }
};

inline float compute_norm(const embedding &a) {
  float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
  for (int i = 0; i < VECTOR_ELEMS; ++i)
    sum += a[i] * a[i];
  return std::sqrt(sum);
}

inline float cosine_similarity(const embedding &a, const embedding &b,
                               float norm_a, float norm_b) {
  float numer = 0.0f;
#pragma omp simd reduction(+ : numer)
  for (int i = 0; i < VECTOR_ELEMS; ++i)
    numer += a[i] * b[i];
  const float denom = norm_a * norm_b;
  return denom == 0.0f ? 0.0f : numer / denom;
}

struct PreparedDatabase {
  const std::vector<embedding> &database;
  std::vector<float> inverse_norms;
  std::vector<std::pair<int, int>> tile_pairs;
};

inline PreparedDatabase prepare_database(const std::vector<embedding> &database) {
  const int num_vectors = static_cast<int>(database.size());
  PreparedDatabase prepared{database, std::vector<float>(num_vectors), {}};

  for (int i = 0; i < num_vectors; ++i) {
    const float norm = compute_norm(database[i]);
    prepared.inverse_norms[i] = norm == 0.0f ? 0.0f : 1.0f / norm;
  }

  const int num_tiles = (num_vectors + TILE_SIZE - 1) / TILE_SIZE;
  prepared.tile_pairs.reserve(num_tiles * (num_tiles + 1) / 2);
  for (int tile_i = 0; tile_i < num_tiles; ++tile_i)
    for (int tile_j = 0; tile_j <= tile_i; ++tile_j)
      prepared.tile_pairs.emplace_back(tile_i, tile_j);

  return prepared;
}

inline float cosine_similarity_from_inverse_norms(const embedding &a,
                                                  const embedding &b,
                                                  float inverse_norm_a,
                                                  float inverse_norm_b) {
  float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
  for (int i = 0; i < VECTOR_ELEMS; ++i)
    dot += a[i] * b[i];
  return dot * inverse_norm_a * inverse_norm_b;
}

// Compute exact top-k cosine-similarity pairs.  Higher scores are more similar.
inline TopKHeap compute_top_k(const PreparedDatabase &prepared) {
  const int num_vectors = static_cast<int>(prepared.database.size());
  TopKHeap final_top_k;

#pragma omp parallel
  {
    TopKHeap local_top_k;

#pragma omp for schedule(dynamic, 1)
    for (int tile_index = 0;
         tile_index < static_cast<int>(prepared.tile_pairs.size()); ++tile_index) {
      const auto [tile_i, tile_j] = prepared.tile_pairs[tile_index];
      const int i_begin = tile_i * TILE_SIZE;
      const int i_end = std::min(i_begin + TILE_SIZE, num_vectors);
      const int j_begin = tile_j * TILE_SIZE;
      const int j_end = std::min(j_begin + TILE_SIZE, num_vectors);

      for (int ii = i_begin; ii < i_end; ++ii) {
        const embedding &a = prepared.database[ii];
        const int jj_end = tile_i == tile_j ? ii : j_end;

        for (int jj = j_begin; jj < jj_end; ++jj) {
          const float score = cosine_similarity_from_inverse_norms(
              a, prepared.database[jj], prepared.inverse_norms[ii],
              prepared.inverse_norms[jj]);
          local_top_k.push(score, ii, jj);
        }
      }
    }

#pragma omp critical
    {
      for (int k = 0; k < local_top_k.size; ++k)
        final_top_k.push(std::get<0>(local_top_k.data[k]),
                         std::get<1>(local_top_k.data[k]),
                         std::get<2>(local_top_k.data[k]));
    }
  }

  return final_top_k;
}

inline TopKHeap compute_top_k(const std::vector<embedding> &database) {
  return compute_top_k(prepare_database(database));
}

inline std::vector<embedding> init(uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::vector<embedding> db(NUM_VECTORS);
  for (auto &vec : db)
    for (float &value : vec)
      value = dis(gen);
  return db;
}

inline std::vector<embedding> init() {
  std::random_device rd;
  return init(rd());
}
