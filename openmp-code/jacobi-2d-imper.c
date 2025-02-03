#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>
/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 20x1000. */
#include "../jacobi-2d-imper.h"


// OPTIMIZATION NOTES
// Created one unique parallel region for the kernel function
// Used collapse(2) in all the nested loops to improve CPU-boundness of the algorithm and see improvements on small datasets
// Created two for directives , with static scheduling to reduce the overheads
// Number of threads can be set in the header file: NUM_THREADS.
// The array updates in the nested loop (A[i][j] and B[i][j]) are independent for each iteration of the outer loop (i).


/* Array initialization. */
static
void init_array (int n,
      DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
      DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  #pragma omp parallel for collapse(2) private(i,j) schedule(static)
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
          A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
          B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
      }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
 void print_array(int n,
      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++){
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


static void kernel_jacobi_2d_imper(int tsteps,
                                   int n,
                                   DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                                   DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int t, i, j;

  for (t = 0; t < tsteps; t++) {
    #pragma omp parallel private(i, j)
    {
      #pragma omp for collapse(2) schedule(static)
      for (i = 1; i < n - 1; i++) {
        for (j = 1; j < n - 1; j++) {
          B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
        }
      }

      #pragma omp for collapse(2) schedule(static)
      for (i = 1; i < n - 1; i++) {
        for (j = 1; j < n - 1; j++) {
          A[i][j] = B[i][j];
        }
      }
    }
  }
}



int main(int argc, char** argv)
{

  omp_set_num_threads(NUM_THREADS);

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d_imper (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed 
  by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
