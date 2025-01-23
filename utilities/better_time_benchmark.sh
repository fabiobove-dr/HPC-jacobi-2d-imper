#!/bin/sh
## Better time_benchmark.sh
## Updated by: Fabio Bove, fabio.bove.dr@gmail.com
## Maximal variance accepted between the 3 median runs for performance results.
## Here 5%
VARIANCE_ACCEPTED=5;

if [ $# -ne 1 ]; then
    echo "Usage: ./time_benchmark_json.sh <binary_name>";
    echo "Example: ./time_benchmark_json.sh \"./a.out\"";
    echo "Note: the file must be a Polybench program compiled with -DPOLYBENCH_TIME";
    exit 1;
fi;

compute_mean_exec_time() {
    file="$1";
    benchcomputed="$2";
    cat "$file" | grep "[0-9]\+" | sort -n | head -n 4 | tail -n 3 > avg.out;
    expr="(0";
    while read n; do
        expr="$expr+$n";
    done < avg.out;
    time=$(echo "scale=8;$expr)/3" | bc);
    tmp=$(echo "$time" | cut -d '.' -f 1);
    if [ -z "$tmp" ]; then
        time="0$time";
    fi;
    val1=$(head -n 1 avg.out);
    val2=$(head -n 2 avg.out | tail -n 1);
    val3=$(head -n 3 avg.out | tail -n 1);
    val11=$(echo "a=$val1 - $time;if(0>a)a*=-1;a" | bc);
    val12=$(echo "a=$val2 - $time;if(0>a)a*=-1;a" | bc);
    val13=$(echo "a=$val3 - $time;if(0>a)a*=-1;a" | bc);
    myvar=$(echo "$val11 $val12 $val13" | awk '{ if ($1 > $2) { if ($1 > $3) print $1; else print $3; } else { if ($2 > $3) print $2; else print $3; } }');
    variance=$(echo "scale=5;($myvar/$time)*100" | bc);
    tmp=$(echo "$variance" | cut -d '.' -f 1);
    if [ -z "$tmp" ]; then
        variance="0$variance";
    fi;
    compvar=$(echo "$variance $VARIANCE_ACCEPTED" | awk '{ if ($1 < $2) print "ok"; else print "error"; }');
    if [ "$compvar" = "error" ]; then
        echo "\033[31m[WARNING]\033[0m Variance is above threshold, unsafe performance measurement";
        echo "        => max deviation=$variance%, tolerance=$VARIANCE_ACCEPTED%";
    else
        echo "[INFO] Maximal deviation from arithmetic mean of 3 average runs: $variance%";
    fi;
    PROCESSED_TIME="$time";
    MAX_VARIANCE="$variance";
    rm -f avg.out;
}

BIN_NAME="$1"
SCRIPT_NAME=$(basename "$1")
RUNS_NUMBER=5
RUN_TIMESTAMP=timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

echo "[INFO] Running $RUNS_NUMBER times $BIN_NAME..."
echo "[INFO] Maximal variance authorized on 3 average runs: $VARIANCE_ACCEPTED%)...";

$BIN_NAME > ____tempfile.data.polybench;
$BIN_NAME >> ____tempfile.data.polybench;
$BIN_NAME >> ____tempfile.data.polybench;
$BIN_NAME >> ____tempfile.data.polybench;
$BIN_NAME >> ____tempfile.data.polybench;

compute_mean_exec_time "____tempfile.data.polybench" "$BIN_NAME";

# Output JSON format
JSON_OUTPUT=$(cat <<EOF
{
    "better-time-evaluation-$RUN_TIMESTAMP": {
        "script-name": "${SCRIPT_NAME}",
        "runs-number": ${RUNS_NUMBER},
        "maximal-variance-authorized": {
            "3-average-runs": "${VARIANCE_ACCEPTED}%"
        },
        "Normalized-time": ${PROCESSED_TIME}
    }
}
EOF
)

# Write JSON output
# if benchmark_result.json exists and is not empty then append JSON output
if [ -s benchmark_result.json ]; then
  # Add comma before appending JSON output
  echo ",\n$JSON_OUTPUT" >> benchmark_result.json
  # Remove square brackets and add them at the beginning and end
  sed -i 's/\[//g; s/\]//g' benchmark_result.json
  sed -i '1s/^/[/' benchmark_result.json && sed -i '$s/$/]/' benchmark_result.json
else
  echo "$JSON_OUTPUT" > benchmark_result.json
fi

echo "[INFO] JSON result written to benchmark_result.json"
rm -f ____tempfile.data.polybench;
