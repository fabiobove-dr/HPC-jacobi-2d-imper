#!/bin/sh

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <executable_path> <gmon_output_path>"
  exit 1
fi

# Assign command-line arguments to variables
EXECUTABLE=$1
GMON_OUT=$2
OUTPUT_FILE="profiling_report.txt"

# Run gprof with the provided paths
gprof "$EXECUTABLE" "$GMON_OUT" > "$OUTPUT_FILE"

# Print success message
echo "Profiling report saved to $OUTPUT_FILE"
