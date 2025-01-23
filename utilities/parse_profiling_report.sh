#!/bin/bash

# Input and Output files
input_file="profiling_report.txt"
output_file="profiling_report.json"

# Initialize JSON output
echo "{" > "$output_file"
echo "  \"flat_profile\": [" >> "$output_file"

# Read the input file and process each line
first=1
while read -r line; do
  # Skip lines without numbers (e.g., comments, headers)
  if [[ "$line" =~ ^[0-9] ]]; then
    # Add a comma to separate the previous object, except for the first one
    if [ "$first" -eq 0 ]; then
      echo "    }," >> "$output_file"
    fi
    first=0

    # Parse the line and extract values
    percentage_time=$(echo "$line" | awk '{print $1}')
    cumulative_seconds=$(echo "$line" | awk '{print $2}')
    self_seconds=$(echo "$line" | awk '{print $3}')
    calls=$(echo "$line" | awk '{print $4}')
    self_time_per_call=$(echo "$line" | awk '{print $5}')
    total_time_per_call=$(echo "$line" | awk '{print $6}')
    function_name=$(echo "$line" | awk '{print $7}')

    # Write the values to the JSON file
    echo "    {" >> "$output_file"
    echo "      \"percentage_time\": \"$percentage_time\"," >> "$output_file"
    echo "      \"cumulative_seconds\": \"$cumulative_seconds\"," >> "$output_file"
    echo "      \"self_seconds\": \"$self_seconds\"," >> "$output_file"
    echo "      \"calls\": \"$calls\"," >> "$output_file"
    echo "      \"self_time_per_call\": \"$self_time_per_call\"," >> "$output_file"
    echo "      \"total_time_per_call\": \"$total_time_per_call\"," >> "$output_file"
    echo "      \"function_name\": \"$function_name\"" >> "$output_file"
  fi
done < "$input_file"

# Close the flat_profile array and JSON
echo "    }" >> "$output_file"
echo "  ]" >> "$output_file"
echo "}" >> "$output_file"

# Display the generated JSON file
cat "$output_file"
