#!/bin/bash -e

# Function to show usage
usage() {
    echo "Usage: $0 executable csv_file dimension values [-j num_cores]"
    echo "  executable   The executable to run (e.g., ./a.out)"
    echo "  csv_file   Input file name (e.g., df_crit.csv)"
    echo "  num_points   Number of points in the input file (e.g., 2188)"
    echo "  dimension    Dimension value (e.g., 3)"
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

# Ensure at least 4 positional arguments are provided
if [ "$#" -lt 5 ]; then
    usage
fi

# Parse positional arguments
executable="$1"
csv_file="$2"
num_points="$3"
dimension="$4"
IFS=',' read -r -a values <<< "$5"
shift 5

# Export variables needed for the command function
export executable
export csv_file
export num_points
export dimension
export values
export SHIFT_TRIES

# Define the function that will run the command
command_function() {
    local value=$1
    python3 ../match_index.py ${csv_file} subset_${value}.txt /dev/null subset_complement_${value}.csv
    python3 ../extract_csv.py subset_complement_${value}.csv subset_complement_${value}.txt
    SHIFT_TRIES=${SHIFT_TRIES:-5000} ${executable} subset_complement_${value}.txt ${dimension} $(expr ${num_points} - ${value}) ${value} subset_complement_subset_${value}.txt 2>&1 | ts > log_subset_complement_${value}.txt
    python3 ../match_index.py ${csv_file} "subset_complement_subset_${value}.txt" "subset_complement_subset_${value}.csv" /dev/null 
    head -n 1 "subset_complement_subset_${value}.txt" | python3 -c 'import sys; import re; print(f"k={sys.argv[1]}", re.search(r"discrepancy=([\d\.]+)", input())[0])' "${value}"
}

# Export the function if parallel execution is used
if [ "$num_cores" -gt 1 ]; then
    export -f command_function
    # Run the commands in parallel with the specified number of cores
    parallel -j "$num_cores" command_function ::: "${values[@]}"
else
    # Run the commands sequentially
    for value in "${values[@]}"; do
        command_function "$value"
    done
fi