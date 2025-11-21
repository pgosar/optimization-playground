CXX      = clang++
CXXFLAGS = -std=c++23
DEBUGFLAGS = -g
TARGET   = main
SRC      = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

debug: $(SRC)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
