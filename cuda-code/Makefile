# Nome del file eseguibile
EXEC = jacobi_2d_cuda

# File sorgente CUDA
SRC = jacobi_2d_cuda.cu

# Compilatore CUDA
NVCC = nvcc

# Flag di compilazione
CFLAGS = -O3 -arch=sm_60  # Ottimizzazione cuda

# Regola di compilazione
$(EXEC): $(SRC)
	$(NVCC) $(CFLAGS) -o $(EXEC) $(SRC)

# Regola per pulire i file compilati
clean:
	rm -f $(EXEC)
