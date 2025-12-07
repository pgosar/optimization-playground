#include <algorithm>
#include <array>
#include <cmath>
#include <omp.h>
#include <random>
#include <tuple>
#include <vector>

#define NUM_VECTORS 8192
#define VECTOR_ELEMS 256
#define K 200
#define TILE_SIZE 128

using embedding = std::array<float, VECTOR_ELEMS>;
using topk_elem = std::tuple<float, int, int>;

struct TopKHeap {
  std::array<topk_elem, K> data{};
  int size = 0;

  void push(const topk_elem &elem) {
    if (size < K) {
      data[size] = elem;
      heapifyUp(size);
      size++;
    } else if (std::get<0>(elem) > std::get<0>(data[0])) {
      data[0] = elem;
      heapifyDown(0);
    }
  }

  void heapifyUp(int idx) {
    while (idx > 0) {
      int parent = (idx - 1) / 2;
      if (std::get<0>(data[parent]) <= std::get<0>(data[idx]))
        break;
      std::swap(data[parent], data[idx]);
      idx = parent;
    }
  }

  void heapifyDown(int idx) {
    while (idx < size) {
      int left = idx * 2 + 1;
      int right = idx * 2 + 2;
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

// Distance functions
inline float compute_norm(const embedding &a) {
  float sum = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++)
    sum += a[i] * a[i];
  return std::sqrt(sum);
}

inline float cosine_similarity(const embedding &a, const embedding &b,
                               float norm_a, float norm_b) {
  float numer = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++)
    numer += a[i] * b[i];
  float denom = norm_a * norm_b;
  return denom == 0.0f ? 0.0f : numer / denom;
}

inline float euclidean_distance(const embedding &a, const embedding &b) {
  float sum = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

inline float manhattan_distance(const embedding &a, const embedding &b) {
  float sum = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++)
    sum += std::fabs(a[i] - b[i]);
  return sum;
}

// Compute a small mode for branch collapsing
inline uint8_t compute_mode(const embedding &a) {
  return (a[0] > 0.5f) + ((a[1] > 0.5f) << 1);
}

// Initialize database
inline std::vector<embedding> init() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::vector<embedding> db(NUM_VECTORS);
  for (auto &vec : db)
    for (float &v : vec)
      v = dis(gen);
  return db;
}

// Compute top-k
TopKHeap compute_top_k(const std::vector<embedding> &database) {
  // Precompute norms and modes
  std::vector<float> norms(NUM_VECTORS);
  std::vector<uint8_t> modes(NUM_VECTORS);
  for (int i = 0; i < NUM_VECTORS; i++) {
    norms[i] = compute_norm(database[i]);
    modes[i] = compute_mode(database[i]);
  }

  TopKHeap final_top_k;

#pragma omp parallel
  {
    TopKHeap local_top_k;

#pragma omp for schedule(dynamic)
    for (int i = 0; i < NUM_VECTORS; i += TILE_SIZE) {
      for (int j = 0; j <= i; j += TILE_SIZE) {
        for (int ii = i; ii < std::min(i + TILE_SIZE, NUM_VECTORS); ii++) {
          const embedding &a = database[ii];
          float norm_a = norms[ii];
          uint8_t mode_a = modes[ii];

          int jj_start = (i == j) ? j : j; // diagonal vs off-diagonal
          int jj_end = std::min(j + TILE_SIZE, NUM_VECTORS);

          for (int jj = jj_start; jj < jj_end; jj++) {
            const embedding &b = database[jj];
            float norm_b = norms[jj];
            uint8_t mode_b = modes[jj];

            uint8_t key = (mode_a << 2) | mode_b;
            float score;

            switch (key) {
            case 0:
              score = manhattan_distance(a, b);
              break;
            case 1:
              score = cosine_similarity(a, b, norm_a, norm_b);
              break;
            case 2:
              score = euclidean_distance(a, b);
              break;
            default:
              score = manhattan_distance(a, b);
              break;
            }

            local_top_k.push({score, ii, jj});
          }
        }
      }
    }

#pragma omp critical
    {
      for (int k = 0; k < local_top_k.size; ++k)
        final_top_k.push(local_top_k.data[k]);
    }
  }

  return final_top_k;
}