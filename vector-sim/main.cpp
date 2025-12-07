#include "lib.hpp"
#include <iostream>
#include <thread>

int main() {
  // so that i have time to attach instruments in Xcode
  std::this_thread::sleep_for(std::chrono::seconds(10));
  auto database = init();
  auto top_k = compute_top_k(database);
  std::cout << "Finished calculating top k\n";
}
