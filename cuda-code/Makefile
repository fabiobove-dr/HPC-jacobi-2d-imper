EXECUTABLE := jacobi-2d-imper
CUFILES := jacobi-2d-imper.cu
PATH_TO_UTILS := ./utilities

all:
	nvcc --ptxas-options=-v -O3 ${CUFILES} -I${PATH_TO_UTILS} -o ${EXECUTABLE}
clean:
	rm -f *~ *.exe