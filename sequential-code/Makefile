CC = gcc
CFLAGS = -pg -O0 -DPOLYBENCH_TIME
SRC = jacobi-2d-imper.c ../utilities/polybench.c

INCLUDE = -I../utilities -Idatadir

all: jacobi-2d-imper

jacobi-2d-imper: $(SRC)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(SRC)

clean:
	rm -f jacobi-2d-imper
