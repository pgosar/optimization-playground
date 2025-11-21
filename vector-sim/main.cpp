#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

// maybe we can cluster vectors by their similarity across various metrics and
// rank pairs of vectors based on how similar the scores for each metric are. we
// can cluster by the top ten lowest, top ten highest, top ten closest to the
// average. Let's also make it so on rare occasions, there can be negative
// numbers in each vector of the database, and these simulate bad values that we
// need to replace with zero

#define NUM_VECTORS 8192
#define VECTOR_ELEMS 128
#define K 200

typedef std::vector<float> embedding;

float cosine_similarity(const embedding &a, const embedding &b) {
  float numer = 0.0f, sum_a = 0.0f, sum_b = 0.0f;
  for (int i = 0; i < VECTOR_ELEMS; i++) {
    numer += a[i] * b[i];
    sum_a += a[i] * a[i];
    sum_b += b[i] * b[i];
  }
  float denom = std::sqrt(sum_a) * std::sqrt(sum_b);
  if (denom == 0.0f)
    return 0.0f;
  return numer / denom;
}

struct Compare {
  bool operator()(const std::tuple<float, embedding, embedding> &a,
                  const std::tuple<float, embedding, embedding> &b) {
    return std::get<0>(a) > std::get<0>(b);
  }
};

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  std::vector<embedding> database(NUM_VECTORS, embedding(VECTOR_ELEMS));
  for (int i = 0; i < NUM_VECTORS; i++)
    for (int j = 0; j < VECTOR_ELEMS; j++)
      database[i][j] = dis(gen);

  std::priority_queue<std::tuple<float, embedding, embedding>,
                      std::vector<std::tuple<float, embedding, embedding>>,
                      Compare>
      top_k;

  for (int i = 0; i < NUM_VECTORS; i++) {
    for (int j = 0; j < i; j++) {
      auto a = database[i];
      auto b = database[j];
      float score = cosine_similarity(a, b);
      std::tuple<float, embedding, embedding> elem(score, a, b);
      if (top_k.size() < K) {
        top_k.push(elem);
      } else if (std::get<0>(top_k.top()) < score) {
        top_k.pop();
        top_k.push(elem);
      }
    }
  }

  std::vector<std::tuple<float, embedding, embedding>> results;
  while (!top_k.empty()) {
    results.push_back(top_k.top());
    top_k.pop();
  }

  std::cout << "Finished calculating top k" << std::endl;
  return 0;
}
