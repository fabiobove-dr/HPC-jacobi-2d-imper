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
Optimize (parallelize) the execution of the assigned
applications on a multiprocessor system.

1. **Analyze** code
2. Finding **hotspots** suitable for parallelization
3. Use **profiling** tools
4. Use the **OpenMP** & **CUDA** programming model
5. Understanding **performance** achieved

---

## Features
- Displays host system and GPU details.
- Refactored Time benchmarking and profiling shell-scripts.
- Different code versions: Sequential, OpenMP and CUDA.

---

## Usage
To compile and run a specific `<version>` of the program:
```bash
cd ./jacobi-2d-imper-<version>
make
./jacobi-2d-imper-<version>
./run.sh
```

---

## Profiling & Time Benchmarking
1. Configure
    - make sure to create a report folder that contains one sub-dir for each "type" of implementation: sequential, cuda, openmp.
    - Edit the header file `jacobi-2d-imper.h` and chose the data size (e.g. `SMALL`, `STANDARD`, `LARGE`, `EXTRA`).
2. Run repetitive tests 
   - In the root you can find the `./do_all_4_me.sh` shell-script.
   - You can run benchmarking as follows: <br> 
   ```bash 
   # e.g../do_all_4_me.sh openmp small 0
   ./do_all_4_me.sh <OPT_TYPE> <DATA_SIZE> <TEST_ONLY>
   ```
3. Reports
- The script will automatically generate a benchmarking report inside the `/report/<OPT_TYPE>` folder.

### Better time benchmarking
Will run automatically 5 x 5 jacobi runs and produce a .json file with the benchmarks of each of the run.

### GNU (binutils tool) identifying hotspots in code.
Will generate a .json file with the report about the gprof execution on the latest gmon.o file produced.

---

### Valgrind (No automatic execution)
To profile the program using Valgrind profiler (works only for sequential-code):
```bash
make clean
make
make profile
kcachegrind callgrind.out.<pid>
```


---
Enjoy 🤓