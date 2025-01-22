![icon](icon.png)


[![Powered by Fabio](https://img.shields.io/badge/Author%20-Fabio%20Bove-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-fabio.bove.dr@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-216219@studenti.unimore.it@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  

--- 

# High Performance Computing - Jacobi 2d Imper
The provided program implements a Jacobi 2D iterative solver in C using the PolyBench library.
It is used to solve a 2D stencil computation.

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