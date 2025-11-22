CXX      = clang++
CXXFLAGS = -std=c++23 -O3 -ffast-math -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
DEBUGFLAGS = -g

# Targets
TARGET   = main
TEST_TARGET = test

# Sources
MAIN_SRC = main.cpp
TEST_SRC = test.cpp
LIB_HDR  = lib.hpp

# Default: build main
all: $(TARGET)

# Build main executable
$(TARGET): $(MAIN_SRC) $(LIB_HDR)
	$(CXX) $(CXXFLAGS) $(MAIN_SRC) -o $(TARGET)

# Build debug version of main
debug: $(MAIN_SRC) $(LIB_HDR)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(MAIN_SRC) -o $(TARGET)

# Build test executable
test: $(TEST_SRC) $(LIB_HDR)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET)

# Run main
run: all
	./$(TARGET)

# Run tests
run-test: test
	./$(TEST_TARGET)

# Clean all binaries
# Clean all binaries
clean:
	rm -rf $(TARGET) $(TEST_TARGET) $(BENCH_TARGET) $(TARGET).dSYM $(TEST_TARGET).dSYM $(BENCH_TARGET).dSYM

# Benchmark
BENCH_TARGET = benchmark
BENCH_SRC = benchmark.cpp

benchmark: $(BENCH_SRC) $(LIB_HDR)
	$(CXX) $(CXXFLAGS) $(BENCH_SRC) -o $(BENCH_TARGET)

run-benchmark: benchmark
	./$(BENCH_TARGET)
