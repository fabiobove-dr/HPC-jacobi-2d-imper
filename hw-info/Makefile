# Makefile for compiling the hw_info program

# Compiler and flags
CC = gcc
CFLAGS = -Wall

# Target executable
TARGET = hw_info

# Source file
SRC = hw_info.c

# Default target to compile the program
all: $(TARGET)

# Rule to compile and link in one step, no object files
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean up generated files (only remove the executable, no object files)
clean:
	rm -f $(TARGET)
