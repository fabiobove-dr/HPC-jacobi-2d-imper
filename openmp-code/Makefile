CC = gcc
CFLAGS = -fopenmp -pg -O0 -DPOLYBENCH_TIME
SRC = jacobi-2d-imper.c ../utilities/polybench.c

INCLUDE = -I../utilities -Idatadir

# Rule to preprocess and remove comments
$(OBJ): $(SRC)
	$(CC) -E $(SRC) | grep -v '^ *#' > $(SRC:.c=.no_comments)
	$(CC) $(CFLAGS) -c $(SRC:.c=.no_comments) -o $(OBJ)

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $< -o $(EXE)

all: jacobi-2d-imper

jacobi-2d-imper: $(SRC)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(SRC)

clean:
	rm -f jacobi-2d-imper
