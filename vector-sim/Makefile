CXX      = clang++
CXXFLAGS = -std=c++23 -O3
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
clean:
	rm -rf $(TARGET) $(TEST_TARGET) $(TARGET).dSYM $(TEST_TARGET).dSYM
