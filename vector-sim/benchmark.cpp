#include "lib.hpp"
#include <chrono>
#include <iostream>

int main() {
  auto database = init();
  const int NUM_RUNS = 50;
  const float INITIAL_TIME = 5.56092;
  double total_time = 0.0;
  TopKHeap top_k;

  std::cout << "Running benchmark " << NUM_RUNS << " times...\n";

  for (int i = 0; i < NUM_RUNS; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    top_k = compute_top_k(database);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    total_time += elapsed.count();
    std::cout << "Run " << (i + 1) << ": " << elapsed.count() << "s\n";
  }
  float avg_time = total_time / NUM_RUNS;
  float improvement = INITIAL_TIME / avg_time;
  std::cout << "Average time: " << avg_time << "s, improvement: " << improvement
            << "x\n";

  std::cout << "Top 1 score: " << std::get<0>(top_k.data[0]) << "\n";
  return 0;
}
