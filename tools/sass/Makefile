CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall
TARGET   = sass_tool
SRCS     = sass.cpp main.cpp

all: $(TARGET)

$(TARGET): $(SRCS) sass.h
	$(CXX) $(CXXFLAGS) -o $@ sass.cpp main.cpp

libsass.a: sass.o
	ar rcs $@ $<

sass.o: sass.cpp sass.h
	$(CXX) $(CXXFLAGS) -c sass.cpp

clean:
	rm -f $(TARGET) libsass.a *.o
