#include "lib.hpp"
#include <iostream>
#include <thread>

int main() {
  // so that i have time to attach instruments in Xcode
  std::this_thread::sleep_for(std::chrono::seconds(10));
  auto database = init();
  std::vector<float> norms(NUM_VECTORS);
  for (int i = 0; i < NUM_VECTORS; i++) {
    norms[i] = compute_norm(database[i]);
  }
  auto top_k = compute_top_k(database, norms);
  std::cout << "Finished calculating top k\n";
}
