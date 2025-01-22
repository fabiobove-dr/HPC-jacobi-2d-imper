# HPC-jacobi-2d-imper
High Performance Computing Project - Jacobi 2d Imper

--- 

## Description

The provided program implements a Jacobi 2D iterative solver in C using the PolyBench library.
It is used to solve a 2D stencil computation.

---
Author:

Fabio Bove | fabio.bove.dr@gmail.com | 216219@studenti.unimore.it

---

## Objective
Optimize (parallelize) the execution time of the assigned
applications on a multiprocessor system.

1. **Analyze** code
2. Finding **hotspots** suitable for parallelization
3. Use **profiling** tools
4. Use the **OpenMP** & **CUDA** programming model
5. Understanding **performance** achieved

---

## Features
- Displays host system and GPU details.

---
## Files
- `jacobi-2d-imper.c`: The benchmark source code.
- `utilities/`: Support utilities for PolyBench benchmarks.
- `datadir/`: Data directory for PolyBench benchmarks.
- `Makefile`: Build configuration.

## Usage
To compile and run the benchmark:
```bash
make
./jacobi-2d-imper-<version>
./utilities/better_time_benchmark.sh ./jacobi-2d-imper-<version>
```

The evaluation produces a `benchmark_result.json` file containing the details on the performance of the code.