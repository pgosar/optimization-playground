#include "lib.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool float_eq(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) < eps;
}

void test_distance_functions() {
  embedding a{};
  embedding b{};
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Cosine similarity
  float norm_a = compute_norm(a);
  float norm_b = compute_norm(b);
  float cos_sim = cosine_similarity(a, b, norm_a, norm_b);
  assert(cos_sim > 0.99f && cos_sim < 1.01f);

  // Euclidean distance
  float euclid = euclidean_distance(a, b);
  float expected = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++)
    expected += (a[i] - b[i]) * (a[i] - b[i]);
  expected = std::sqrt(expected);
  assert(float_eq(euclid, expected));

  // Manhattan distance
  float manh = manhattan_distance(a, b);
  expected = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++)
    expected += std::fabs(a[i] - b[i]);
  assert(float_eq(manh, expected));

  std::cout << "Distance function tests passed.\n";
}

void assert_minheap(const TopKHeap &h) {
  for (int i = 0; i < h.size; i++) {
    int left = i * 2 + 1;
    int right = i * 2 + 2;

    if (left < h.size) {
      assert(std::get<0>(h.data[i]) <= std::get<0>(h.data[left]));
    }
    if (right < h.size) {
      assert(std::get<0>(h.data[i]) <= std::get<0>(h.data[right]));
    }
  }
}

void test_topkheap() {
  TopKHeap heap;

  for (int i = 1000; i >= 950; i--) {
    heap.push({float(i), i, i});
    assert_minheap(heap);
  }
  assert(heap.size == 51);

  float smallest = std::get<0>(heap.data[0]);
  for (int i = 1; i < heap.size; i++)
    assert(std::get<0>(heap.data[i]) >= smallest);

  for (int i = heap.size; i < K; i++) {
    heap.push({float(i + 1), i, i});
    assert_minheap(heap);
  }
  assert(heap.size == K);

  for (int i = 5000; i < 5100; i++) {
    heap.push({float(i), i, i});
    assert_minheap(heap);

    float root = std::get<0>(heap.data[0]);
    for (int j = 1; j < K; j++)
      assert(std::get<0>(heap.data[j]) >= root);
  }

  float current_min = std::get<0>(heap.data[0]);

  heap.push({current_min - 10.0f, 12345, 54321});
  assert(std::get<0>(heap.data[0]) == current_min); // unchanged
  assert_minheap(heap);

  heap.push({current_min + 1000.0f, 999, 999});
  assert(std::get<0>(heap.data[0]) != current_min);
  assert_minheap(heap);

  {
    TopKHeap random_heap;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);

    for (int i = 0; i < 5000; i++) {
      float val = dis(rng);
      random_heap.push({val, i, i});
      assert_minheap(random_heap);
    }
    assert(random_heap.size == K);
    assert_minheap(random_heap);
  }

  std::cout << "TopKHeap tests passed.\n";
}

void test_init() {
  auto db = init();
  assert(db.size() == NUM_VECTORS);
  for (auto &vec : db)
    assert(vec.size() == VECTOR_ELEMS);
  std::cout << "Init function test passed.\n";
}

float auto_metric(const embedding &a, const embedding &b, float norm_a, float norm_b) {
  if (a[0] > b[0])
    return cosine_similarity(a, b, norm_a, norm_b);
  else if (a[1] + b[1] > 1.0f)
    return euclidean_distance(a, b);
  else
    return manhattan_distance(a, b);
}

void test_compute_top_k() {
  std::vector<embedding> db(NUM_VECTORS);

  // Use random data for better coverage (fixed seed for reproducibility)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (auto &vec : db)
    for (float &v : vec)
      v = dis(rng);

  std::vector<float> norms(NUM_VECTORS);
  for (int i = 0; i < NUM_VECTORS; i++) {
    norms[i] = compute_norm(db[i]);
  }

  struct R {
    float score;
    int i, j;
  };
  std::vector<R> all_pairs;

  // generate ALL ground truth pairs using *same metric logic*
  for (int i = 0; i < NUM_VECTORS; i++) {
    for (int j = 0; j < i; j++) {
      float s = auto_metric(db[i], db[j], norms[i], norms[j]);
      all_pairs.push_back({s, i, j});
    }
  }

  std::sort(all_pairs.begin(), all_pairs.end(), [](const R &a, const R &b) {
    return a.score > b.score; // descending
  });

  std::vector<R> expected(all_pairs.begin(), all_pairs.begin() + K);

  // actual result
  auto top_k = compute_top_k(db);

  std::vector<R> actual;
  actual.reserve(K);
  for (int i = 0; i < K; i++)
    actual.push_back({std::get<0>(top_k.data[i]), std::get<1>(top_k.data[i]),
                      std::get<2>(top_k.data[i])});

  // sort actual so ordering doesn't matter
  std::sort(actual.begin(), actual.end(),
            [](const R &a, const R &b) { return a.score > b.score; });

  for (int i = 0; i < K; i++) {
    if (!float_eq(actual[i].score, expected[i].score)) {
      std::cout << "Score mismatch at rank " << i
                << ": actual=" << actual[i].score
                << " expected=" << expected[i].score << "\n";
      assert(false);
    }
  }

  std::cout << "top-k test passed.\n";
}

int main() {
  test_distance_functions();
  test_topkheap();
  test_init();
  test_compute_top_k();
  std::cout << "All tests passed successfully!\n";
  return 0;
}
