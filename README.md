![icon](icon.png)
[![Powered by Fabio](https://img.shields.io/badge/Author%20-Fabio%20Bove-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-fabio.bove.dr@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-216219@studenti.unimore.it@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  

--- 

# High Performance Computing - Jacobi 2d Imper
The provided program implements a Jacobi 2D iterative solver in C using the PolyBench library.
It is used to solve a 2D stencil computation.

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
- Refactored Time benchmarking shell-script.
- Simple profiling tool using GNU.

---

## Usage
To compile and run the time benchmarking on a specific `<version>` of the program:
```bash
cd ./jacobi-2d-imper-<version>
make
./jacobi-2d-imper-<version>
./utilities/better_time_benchmark.sh ./jacobi-2d-imper-<version>
```

The evaluation produces a file (`benchmark_result.json`) containing the details on the performance of the code.

---
How many operations I must do?
• How it scales?
How much data I must access?
• Which data layout?
• 2D, 3D, sparse?
• Double, float, int
Which function costs more?
• Where are my hot spots?
• Where I spent more time?

---

## Profiling & Time Benchmarking
### Better time benchmarking
1. Edit the `.h` file of `jacobi-2d-imper` and chose the data size (e.g. LARGE, STANDARD, SMALL, ...).
2. Go to folder containing `jacobi-2d-imper.c` code of interest, run `make clean` - `make`.
3. Go to `./utilities` folder.
4. Run the `better_time_benchmarking.sh` script with arg `../<program-method>/jacobi-2d-imper `(e.g `../sequential-code/jacobi-2d-imper`). **Note**: RUN it a few times to collect some data.
5. Copy the `benchmark_result.json` file, that has been generated, to the `./report` folder and rename it as `<size_type>` (e.g. `standard_sequential` or `extra_large_sequential`). Do this for each data size report.
6. Run the `benchmark_analysis` notebook to generate some plots.

### GNU (binutils tool) identifying hotspots in code.
- The profiling is enabled by default while compiling the code (`-pg`)
- Execute the profiling by running the script 
```bash
./utilities/run_profiling.sh`
```
- To parse the output in a more readable format use
```bash
./utilities/parse_profiling_report.sh
```
### How to read the `json` report

- `percentage_time`: The percentage of the total program time spent in a particular function. A value of "100.00" indicates that this function used 100% of the program's execution time, while "0.00" indicates a negligible contribution to the total time. 
- `cumulative_seconds`: The percentage of the total program time spent in a particular function. A value of "100.00" indicates that this function used 100% of the program's execution time, while "0.00" indicates a negligible contribution to the total time.
- `self_seconds`: The time spent in the function itself, excluding time spent in its child functions.
- `calls`: The number of times this function was called during the program's execution.
- `self_time_per_call`: The average self time for each function call. This is the time the function took per invocation, excluding time spent in called functions.
- `total_time_per_call`:  The total time taken per function call, including time spent in called functions. This field seems to be empty in your example, which could indicate that the measurement for this metric wasn't available or relevant.
- `function_name`:  The name of the function being reported. If this is empty, the function is possibly an internal system function or an aggregate for multiple functions.

### Valgrind
To profile the program using Valgrind profiler:
```bash
make clean
make
make profile
kcachegrind callgrind.out.<pid>
```


