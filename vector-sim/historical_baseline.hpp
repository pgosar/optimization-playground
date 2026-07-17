#pragma once

#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

// Faithful compute-path baseline from the initial commit.  It intentionally
// keeps the dynamic embeddings, per-pair copies, and heap entries containing
// whole vectors so benchmark.cpp can measure the real historical baseline.
constexpr int HISTORICAL_NUM_VECTORS = 8192;
constexpr int HISTORICAL_VECTOR_ELEMS = 128;
constexpr int HISTORICAL_K = 200;

using historical_embedding = std::vector<float>;
using historical_topk_elem =
    std::tuple<float, historical_embedding, historical_embedding>;

struct HistoricalCompare {
  bool operator()(const historical_topk_elem &a,
                  const historical_topk_elem &b) const {
    return std::get<0>(a) > std::get<0>(b);
  }
};

float historical_cosine_similarity(const historical_embedding &a,
                                   const historical_embedding &b);
std::vector<historical_embedding> historical_init(uint32_t seed);
float historical_compute_top_k(
    const std::vector<historical_embedding> &database);
