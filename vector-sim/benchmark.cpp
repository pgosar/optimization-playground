#include "lib.hpp"
#include "historical_baseline.hpp"
#include <chrono>
#include <iostream>
#include <vector>

struct TimingSummary {
  double average;
  double minimum;
  double median;
  double p95;
  double maximum;
};

TimingSummary summarize(std::vector<double> samples) {
  double total = 0.0;
  for (double sample : samples)
    total += sample;
  std::sort(samples.begin(), samples.end());
  return {total / samples.size(), samples.front(), samples[samples.size() / 2],
          samples[(samples.size() * 95) / 100], samples.back()};
}

void print_summary(const char *label, const TimingSummary &summary) {
  std::cout << label << " — average: " << summary.average
            << "s, min: " << summary.minimum << "s, median: "
            << summary.median << "s, p95: " << summary.p95
            << "s, max: " << summary.maximum << "s\n";
}

int main() {
  constexpr uint32_t benchmark_seed = 42;
  constexpr int comparison_runs = 5;
  auto database = init(benchmark_seed);
  auto prepared_database = prepare_database(database);
  std::vector<double> samples;
  samples.reserve(comparison_runs);
  TopKHeap top_k;

  // Create the OpenMP team and warm caches before collecting measurements.
  compute_top_k(prepared_database);
  std::cout << "Running " << comparison_runs
            << " optimized timed iterations...\n";

  for (int i = 0; i < comparison_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    top_k = compute_top_k(prepared_database);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    samples.push_back(elapsed.count());
  }

  const TimingSummary optimized = summarize(samples);
  print_summary("Optimized 256D cosine search (prepared)", optimized);

  auto historical_database = historical_init(benchmark_seed);
  std::vector<double> historical_samples;
  historical_samples.reserve(comparison_runs);
  float historical_top_score = 0.0f;
  std::cout << "Running " << comparison_runs
            << " historical-baseline iterations...\n";
  for (int i = 0; i < comparison_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    historical_top_score = historical_compute_top_k(historical_database);
    auto end = std::chrono::high_resolution_clock::now();
    historical_samples.push_back(
        std::chrono::duration<double>(end - start).count());
  }

  const TimingSummary historical = summarize(historical_samples);
  print_summary("Historical 128D cosine workload", historical);
  std::cout << "Historical-workload ratio (not normalized for dimension or build flags): "
            << historical.average / optimized.average << "x\n";

  std::cout << "Optimized best score: " << top_k.best_score()
            << ", optimized top-k threshold: " << top_k.threshold()
            << ", historical top-k threshold: " << historical_top_score << "\n";
  return 0;
}
