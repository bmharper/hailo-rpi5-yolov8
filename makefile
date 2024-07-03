# Compiler
CXX = g++

# Compiler flags

# Release
CXXFLAGS = -std=c++17 -O3

# Debug
#CXXFLAGS = -std=c++17 -g -O0

LDFLAGS = -lhailort

# Source files
SRCS = yolov8.cpp allocator.cpp

# Object files directory
OBJDIR = bin

# Object files
OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

# Executable name
TARGET = $(OBJDIR)/yolohailo

# Default target
all: $(TARGET)

# Link the object files to create the final executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile each source file into an object file
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure the object directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean up the build directory
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
