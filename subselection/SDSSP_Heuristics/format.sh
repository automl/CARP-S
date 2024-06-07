#!/bin/bash -e

usage() {
    echo "Usage: $0 csv_file values [-j num_cores]"
    echo "  csv_file     The CSV file to process (e.g., df_crit.csv)"
    echo "  values       A comma-separated list of values (number of points in the subset) (e.g., 40,60,80,90)"
    echo "  -j num_cores (Optional) Number of cores to use in parallel (default is 1)"
    exit 1
}

# Default number of cores
num_cores=1

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--num_cores)
      num_cores="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      usage
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Ensure at least 3 positional arguments are provided
if [ "$#" -lt 2 ]; then
    usage
fi

# Parse positional arguments
csv_file="$1"
IFS=',' read -r -a values <<< "$2"
shift 2

export csv_file

# Define the function that will run the command
command_function() {
    local value=$1
    python3 ../match_index.py "$csv_file" "subset_${value}.txt" "subset_${value}.csv" /dev/null 
    head -n 1 "subset_${value}.txt" | python3 -c 'import sys; import re; print(f"k={sys.argv[1]}", re.search(r"discrepancy=([\d\.]+)", input())[0])' "${value}"
}

# Export the function to be used in parallel
export -f command_function

# Run the commands either in parallel or sequentially based on num_cores
if [ "$num_cores" -gt 1 ]; then
    parallel -j "$num_cores" command_function ::: "${values[@]}"
else
    for value in "${values[@]}"; do
        command_function "$value"
    done
fi