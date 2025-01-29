#include <stdio.h>
#include <cuda.h>

#define N 1000  // Numero di righe della matrice
#define M 1000  // Numero di colonne della matrice
#define BLOCK_SIZE 16  // Dimensione del blocco per CUDA

typedef double DATA_TYPE;

// Kernel CUDA per moltiplicare ogni elemento della matrice per 0.2
__global__ void multiply_kernel(DATA_TYPE* A, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        A[i * m + j] *= 0.2;
    }
}

// Inizializza la matrice con valori predefiniti
void init_matrix(int n, int m, DATA_TYPE* A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i * m + j] = (DATA_TYPE)(i + j);
        }
    }
}

// Stampa la matrice
void print_matrix(int n, int m, DATA_TYPE* A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            fprintf(stderr,"%.2f ", A[i * m + j]);
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}

int main() {
    int n = N;
    int m = M;
    size_t size = n * m * sizeof(DATA_TYPE);

    // Allocazione della memoria sulla CPU (host)
    DATA_TYPE *h_A = (DATA_TYPE*)malloc(size);

    // Inizializza la matrice
    init_matrix(n, m, h_A);

    // Allocazione della memoria sulla GPU (device)
    DATA_TYPE *d_A;
    cudaMalloc((void**)&d_A, size);

    // Copia i dati dalla CPU alla GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Configura la griglia e i blocchi
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Esegue il kernel CUDA
    multiply_kernel<<<grid, block>>>(d_A, n, m);

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // Stampa la matrice risultante
    //print_matrix(n, m, h_A);

    // Libera la memoria allocata sulla CPU e sulla GPU
    free(h_A);
    cudaFree(d_A);

    return 0;
    //sudo apt install nvidia-cuda-toolkit
    //nvcc -o test test.cu
}