CC = gcc-13
CFLAGS = -fopenmp -O3 -Wall
TARGET = libaco_standard.so
SRC = aco.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -shared -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

