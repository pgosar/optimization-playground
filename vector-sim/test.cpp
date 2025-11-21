#include "lib.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool float_eq(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) < eps;
}

// Unit tests for distance functions
void test_distance_functions() {
  embedding a{};
  embedding b{};
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Cosine similarity
  float cos_sim = cosine_similarity(a, b);
  // Allow for floating-point precision: cosine similarity should be close to 1
  assert(cos_sim > 0.99f && cos_sim < 1.01f); // Should be close to 1 since vectors are positively correlated

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

// Unit tests for TopKHeap
void test_topkheap() {
  TopKHeap heap;

  // Add less than K elements
  for (int i = 0; i < K / 2; i++)
    heap.push({float(i), i, i});

  assert(heap.size == K / 2);

  // Add elements to exceed K
  for (int i = K / 2; i < K + 50; i++)
    heap.push({float(i), i, i});

  assert(heap.size == K);

  // The smallest element in the heap should be at root
  float min_val = std::get<0>(heap.data[0]);
  for (int i = 1; i < K; i++)
    assert(std::get<0>(heap.data[i]) >= min_val);

  std::cout << "TopKHeap tests passed.\n";
}

// Test init function
void test_init() {
  auto db = init();
  assert(db.size() == NUM_VECTORS);
  for (auto &vec : db)
    assert(vec.size() == VECTOR_ELEMS);
  std::cout << "Init function test passed.\n";
}

// End-to-end test for compute_top_k
void test_compute_top_k() {
  std::vector<embedding> db(NUM_VECTORS);
  for (int i = 0; i < NUM_VECTORS; i++)
    for (int j = 0; j < VECTOR_ELEMS; j++)
      db[i][j] = 1.0f; // All vectors identical

  auto top_k = compute_top_k(db);

  // All scores should be consistent (cosine similarity = 1 for identical
  // vectors)
  for (int i = 0; i < K; i++)
    assert(float_eq(std::get<0>(top_k.data[i]), 1.0f) ||
           std::get<0>(top_k.data[i]) == 0.0f);

  std::cout << "End-to-end top-k test passed.\n";
}

int main() {
  test_distance_functions();
  test_topkheap();
  test_init();
  test_compute_top_k();
  std::cout << "All tests passed successfully!\n";
  return 0;
}
