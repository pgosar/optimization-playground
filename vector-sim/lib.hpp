#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#define NUM_VECTORS 8192
#define VECTOR_ELEMS 256
#define K 200

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
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    sum += a[i] * a[i];
  }
  return std::sqrt(sum);
}

inline float cosine_similarity(const embedding &a, const embedding &b, float norm_a, float norm_b) {
  float numer = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    numer += a[i] * b[i];
  }
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
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    sum += std::fabs(a[i] - b[i]);
  }
  return sum;
}

// Initialization
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
inline TopKHeap compute_top_k(const std::vector<embedding> &database, const std::vector<float> &norms) {
  TopKHeap final_top_k;

  // Tiling parameters
  const int TILE_SIZE = 128;

#pragma omp parallel
  {
    TopKHeap top_k;

#pragma omp for schedule(dynamic)
    for (int i = 0; i < NUM_VECTORS; i += TILE_SIZE) {
      // Square blocks
      for (int j = 0; j < i; j += TILE_SIZE) {
        for (int ii = i; ii < i + TILE_SIZE; ii++) {
          const embedding &a = database[ii];
          for (int jj = j; jj < j + TILE_SIZE; jj++) {
            const embedding &b = database[jj];
            float score;
            if (a[0] > b[0])
              score = cosine_similarity(a, b, norms[ii], norms[jj]);
            else if (a[1] + b[1] > 1.0f)
              score = euclidean_distance(a, b);
            else
              score = manhattan_distance(a, b);

            top_k.push({score, ii, jj});
          }
        }
      }

      // Diagonal block
      for (int ii = i; ii < i + TILE_SIZE; ii++) {
        const embedding &a = database[ii];
        for (int jj = i; jj < ii; jj++) {
          const embedding &b = database[jj];

          float score;
          if (a[0] > b[0])
            score = cosine_similarity(a, b, norms[ii], norms[jj]);
          else if (a[1] + b[1] > 1.0f)
            score = euclidean_distance(a, b);
          else
            score = manhattan_distance(a, b);

          top_k.push({score, ii, jj});
        }
      }
    }

#pragma omp critical
    {
      // Merge local heap into final heap
      for (int k = 0; k < top_k.size; ++k) {
        final_top_k.push(top_k.data[k]);
      }
    }
  }

  return final_top_k;
}
