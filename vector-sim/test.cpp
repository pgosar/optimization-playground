#include "lib.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool float_eq(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) < eps;
}

void test_cosine_similarity() {
  embedding a{};
  embedding b{};
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  float norm_a = compute_norm(a);
  float norm_b = compute_norm(b);
  float cos_sim = cosine_similarity(a, b, norm_a, norm_b);
  assert(cos_sim > 0.99f && cos_sim < 1.01f);

  embedding zero{};
  assert(cosine_similarity(a, zero, norm_a, compute_norm(zero)) == 0.0f);

  std::cout << "Cosine similarity tests passed.\n";
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
    heap.push(float(i), i, i);
    assert_minheap(heap);
  }
  assert(heap.size == 51);

  float smallest = std::get<0>(heap.data[0]);
  for (int i = 1; i < heap.size; i++)
    assert(std::get<0>(heap.data[i]) >= smallest);

  for (int i = heap.size; i < K; i++) {
    heap.push(float(i + 1), i, i);
    assert_minheap(heap);
  }
  assert(heap.size == K);

  for (int i = 5000; i < 5100; i++) {
    heap.push(float(i), i, i);
    assert_minheap(heap);

    float root = std::get<0>(heap.data[0]);
    for (int j = 1; j < K; j++)
      assert(std::get<0>(heap.data[j]) >= root);
  }

  float current_min = std::get<0>(heap.data[0]);

  heap.push(current_min - 10.0f, 12345, 54321);
  assert(std::get<0>(heap.data[0]) == current_min); // unchanged
  assert_minheap(heap);

  heap.push(current_min + 1000.0f, 999, 999);
  assert(std::get<0>(heap.data[0]) != current_min);
  assert_minheap(heap);

  {
    TopKHeap random_heap;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);

    for (int i = 0; i < 5000; i++) {
      float val = dis(rng);
      random_heap.push(val, i, i);
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

  // Keep an independent, sequential streaming top-k oracle.  Sorting every
  // pair creates a large allocation and makes the test much slower than the
  // algorithm it validates.
  TopKHeap expected;
  for (int i = 0; i < NUM_VECTORS; i++) {
    for (int j = 0; j < i; j++) {
      expected.push(cosine_similarity(db[i], db[j], norms[i], norms[j]), i,
                    j);
    }
  }

  // actual result
  const auto prepared = prepare_database(db);
  auto top_k = compute_top_k(prepared);

  auto descending = [](const topk_elem &a, const topk_elem &b) {
    return std::get<0>(a) > std::get<0>(b);
  };
  std::sort(expected.data.begin(), expected.data.begin() + expected.size,
            descending);
  std::sort(top_k.data.begin(), top_k.data.begin() + top_k.size, descending);

  for (int i = 0; i < K; i++) {
    if (!float_eq(std::get<0>(top_k.data[i]),
                  std::get<0>(expected.data[i]))) {
      std::cout << "Score mismatch at rank " << i
                << ": actual=" << std::get<0>(top_k.data[i])
                << " expected=" << std::get<0>(expected.data[i]) << "\n";
      assert(false);
    }
  }

  std::cout << "top-k test passed.\n";
}

int main() {
  test_cosine_similarity();
  test_topkheap();
  test_init();
  test_compute_top_k();
  std::cout << "All tests passed successfully!\n";
  return 0;
}
