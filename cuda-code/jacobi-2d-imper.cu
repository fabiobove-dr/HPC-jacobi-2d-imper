#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#define POLYBENCH_TIME 1

#include "jacobi-2d-imper.cuh"
#include <polybench.h>
#include <polybench.c>


//OPTIMIZATION NOTES:
// Creted two kernels: compute_vals and store_vals
// Run kernel, split into two parts: Compute the values of h_B, store the values of h_B
// one iteration is done by the composition of both kernels
// Switched to cudaDeviceSynchronize, instead of cudaThreadSynchronize  which is now deprecated for v9 and later

void
init_array(int n, DATA_TYPE
    POLYBENCH_2D(h_A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(h_B,N,N,n,n))
{
	int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      h_A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      h_B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

__global__ void
kernel_jacobi_2d_imper_compute_vals(int n, DATA_TYPE* h_A, DATA_TYPE* h_B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

    // NB: The stencil computation requires neighboring elements, so we must avoid the first (i == 0, j == 0) and last (i == N-1, j == N-1) rows and columns.
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		h_B[i*N + j] = 0.2f * (
		    h_A[i*N + j] +
		    h_A[i*N + (j-1)] +
		    h_A[i*N + (1 + j)] +
		    h_A[(1 + i)*N + j]
		    + h_A[(i-1)*N + j]
		);
    }
}


__global__ void
kernel_jacobi_2d_imper_store_vals(int n, DATA_TYPE* h_A, DATA_TYPE* h_B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		h_A[i*N + j] = h_B[i*N + j];
	}
}


void
runJacobi2DCUDA(int tsteps, int n,
    DATA_TYPE POLYBENCH_2D(h_A,N,N,n,n), 
    DATA_TYPE POLYBENCH_2D(h_B,N,N,n,n), 
    DATA_TYPE POLYBENCH_2D(d_A_out,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(d_B_out,N,N,n,n))
{
	DATA_TYPE* d_A;
	DATA_TYPE* d_B;

	cudaMalloc(&d_A, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&d_B, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(d_A, h_A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
	
	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		kernel_jacobi_2d_imper_compute_vals<<<grid,block>>>(n, d_A, d_B);
		cudaDeviceSynchronize();
		kernel_jacobi_2d_imper_store_vals<<<grid,block>>>(n, d_A, d_B);
		cudaDeviceSynchronize();
	}

	/* Stop and print timer. */
  	polybench_stop_instruments;
  	polybench_print_instruments;

	// Copy result from GPU to CPU
	cudaMemcpy(d_A_out, d_A, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_B_out, d_B, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

    // free device memory
	cudaFree(d_A);
	cudaFree(d_B);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(h_A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, h_A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

    /* Variable declaration/allocation. */
    // Now we have both host and device arrays
	POLYBENCH_2D_ARRAY_DECL(h_a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(h_b,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(d_a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(d_b,DATA_TYPE,N,N,n,n);

    /* Initialize array(s). */
	init_array(n, POLYBENCH_ARRAY(h_a), POLYBENCH_ARRAY(h_b));

	/* run the cuda kernels */
	runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(h_a), POLYBENCH_ARRAY(h_b), POLYBENCH_ARRAY(d_a), POLYBENCH_ARRAY(d_b));

	polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(d_a)));

    // Free memory on device and host
	POLYBENCH_FREE_ARRAY(h_a);
	POLYBENCH_FREE_ARRAY(h_b);
	POLYBENCH_FREE_ARRAY(d_a);
	POLYBENCH_FREE_ARRAY(d_b);

	return 0;
}

