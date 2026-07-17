#include "historical_baseline.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  constexpr uint32_t benchmark_seed = 42;
  constexpr int historical_baseline_runs = 5;
  const auto database = historical_init(benchmark_seed);
  std::vector<double> samples;
  samples.reserve(historical_baseline_runs);
  float top_score = 0.0f;

  std::cout << "Running " << historical_baseline_runs
            << " historical-baseline iterations...\n";
  for (int i = 0; i < historical_baseline_runs; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    top_score = historical_compute_top_k(database);
    const auto end = std::chrono::high_resolution_clock::now();
    samples.push_back(std::chrono::duration<double>(end - start).count());
  }

  double total = 0.0;
  for (double sample : samples)
    total += sample;
  std::sort(samples.begin(), samples.end());
  std::cout << "Historical baseline — average: "
            << total / historical_baseline_runs
            << "s, min: " << samples.front()
            << "s, median: " << samples[historical_baseline_runs / 2]
            << "s, p95: "
            << samples[(historical_baseline_runs * 95) / 100]
            << "s, max: " << samples.back()
            << "s, top 1 score: " << top_score << "\n";
  return 0;
}
