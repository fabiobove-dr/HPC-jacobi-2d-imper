#!/bin/bash

# Ensure the script exits on errors and undefined variables
set -euo pipefail

# Check for required arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <OPT_TYPE> <DATA_SIZE> <TEST_ONLY>"
  exit 1
fi

# Assign arguments to variables
TYPE_ARG_CLEAN=$1""
TYPE_ARG=$1"-code"
SIZE_ARG=$2
TEST_ONLY=$3

# Print arguments
echo "Running with TYPE_ARG=$TYPE_ARG, SIZE_ARG=$SIZE_ARG, TEST_ONLY=$TEST_ONLY"

#####################################################################################################################################################################
# Ensure the report directory exists before copying
REPORT_DIR="../report"
mkdir -p "$REPORT_DIR"

#####################################################################################################################################################################
# Compile and run the program
if [ "$TEST_ONLY" -eq 0 ]; then
  cd ./"$TYPE_ARG" || { echo "Error: Directory '$TYPE_ARG' not found."; exit 1; }
  ./run.sh >/dev/null
  cd ..
fi

#####################################################################################################################################################################
cd ./utilities || { echo "Error: Directory 'utilities' not found."; exit 1; }
# Profiling
# ./run_profiling.sh ../"$TYPE_ARG"/jacobi-2d-imper ../"$TYPE_ARG"/gmon.out > /dev/null
# ./parse_profiling_report.sh > /dev/null
# cp profiling_report.json "$REPORT_DIR/profiling_${SIZE_ARG}_${TYPE_ARG_CLEAN}.json"

# Time Benchmarking
for _ in {1..5}; do
 ./better_time_benchmark.sh ../"$TYPE_ARG"/jacobi-2d-imper >/dev/null
done
# Copy benchmark result to the appropriate file
cp benchmark_result.json "$REPORT_DIR/${TYPE_ARG_CLEAN}/${SIZE_ARG}_${TYPE_ARG_CLEAN}.json"

#####################################################################################################################################################################
echo "Script completed successfully. Benchmark and profiling results saved to ./${REPORT_DIR}"
rm gmon.out
rm benchmark_result.json
rm profiling_report.json
rm profiling_report.txt