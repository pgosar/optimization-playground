#include "historical_baseline.hpp"

float historical_cosine_similarity(const historical_embedding &a,
                                   const historical_embedding &b) {
  float numer = 0.0f;
  float sum_a = 0.0f;
  float sum_b = 0.0f;
  for (int i = 0; i < HISTORICAL_VECTOR_ELEMS; ++i) {
    numer += a[i] * b[i];
    sum_a += a[i] * a[i];
    sum_b += b[i] * b[i];
  }
  const float denom = std::sqrt(sum_a) * std::sqrt(sum_b);
  return denom == 0.0f ? 0.0f : numer / denom;
}

std::vector<historical_embedding> historical_init(uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::vector<historical_embedding> database(
      HISTORICAL_NUM_VECTORS,
      historical_embedding(HISTORICAL_VECTOR_ELEMS));
  for (auto &vector : database)
    for (float &value : vector)
      value = dis(gen);
  return database;
}

float historical_compute_top_k(
    const std::vector<historical_embedding> &database) {
  std::priority_queue<historical_topk_elem, std::vector<historical_topk_elem>,
                      HistoricalCompare>
      top_k;

  for (int i = 0; i < HISTORICAL_NUM_VECTORS; ++i) {
    for (int j = 0; j < i; ++j) {
      auto a = database[i];
      auto b = database[j];
      const float score = historical_cosine_similarity(a, b);
      historical_topk_elem elem(score, a, b);
      if (top_k.size() < HISTORICAL_K) {
        top_k.push(elem);
      } else if (std::get<0>(top_k.top()) < score) {
        top_k.pop();
        top_k.push(elem);
      }
    }
  }

  return std::get<0>(top_k.top());
}
