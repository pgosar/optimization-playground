CXX      = clang++
CXXFLAGS = -std=c++23 -O3 -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
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
	rm -rf $(TARGET) $(TEST_TARGET) $(BENCH_TARGET) $(HISTORICAL_TARGET) $(HISTORICAL_OBJ) $(TARGET).dSYM $(TEST_TARGET).dSYM $(BENCH_TARGET).dSYM $(HISTORICAL_TARGET).dSYM

# Benchmark
BENCH_TARGET = vector-sim-benchmark
BENCH_SRC = benchmark.cpp
BENCH_HDR = $(LIB_HDR) historical_baseline.hpp
HISTORICAL_TARGET = historical-benchmark
HISTORICAL_SRC = historical_benchmark.cpp
HISTORICAL_LIB_SRC = historical_baseline.cpp
HISTORICAL_HDR = historical_baseline.hpp
HISTORICAL_OBJ = historical_baseline.o
HISTORICAL_CXXFLAGS = -std=c++23

benchmark: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_SRC) $(BENCH_HDR) $(HISTORICAL_OBJ)
	$(CXX) $(CXXFLAGS) $(BENCH_SRC) $(HISTORICAL_OBJ) -o $(BENCH_TARGET)

# The initial commit used only -std=c++23: no -O3, -ffast-math, or OpenMP.
$(HISTORICAL_OBJ): $(HISTORICAL_LIB_SRC) $(HISTORICAL_HDR)
	$(CXX) $(HISTORICAL_CXXFLAGS) -c $(HISTORICAL_LIB_SRC) -o $(HISTORICAL_OBJ)

historical: $(HISTORICAL_TARGET)

$(HISTORICAL_TARGET): $(HISTORICAL_SRC) $(HISTORICAL_HDR) $(HISTORICAL_OBJ)
	$(CXX) $(HISTORICAL_CXXFLAGS) $(HISTORICAL_SRC) $(HISTORICAL_OBJ) -o $(HISTORICAL_TARGET)

run-benchmark: benchmark
	./$(BENCH_TARGET)

.PHONY: benchmark historical compare run-benchmark run-historical

# Build and run a five-run optimized-versus-historical comparison.
compare: benchmark
	./$(BENCH_TARGET)

run-historical: historical
	./$(HISTORICAL_TARGET)
