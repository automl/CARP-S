#!/bin/bash -e

# Function to show usage
usage() {
    echo "Usage: $0 executable input_file dimension num_points values [-j num_cores]"
    echo "  executable   The executable to run (e.g., ./a.out)"
    echo "  input_file   Input file name (e.g., df_crit.txt)"
    echo "  dimension    Dimension value (e.g., 3)"
    echo "  num_points   Number of points in the input file (e.g., 2188)"
    echo "  values       A comma-separated list of values (number of points in the subset) (e.g., 40,60,80,90)"
    echo "  -j num_cores (Optional) Number of cores to use in parallel (default is 1)"
    exit 1
}


# Default number of cores
num_cores=1

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--num-cores)
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

# Ensure at least 5 positional arguments are provided
if [ "$#" -lt 5 ]; then
    usage
fi

# Parse positional arguments
executable="$1"
input_file="$2"
dimension="$3"
num_points="$4"
IFS=',' read -r -a values <<< "$5"
shift 5

export executable
export input_file
export dimension    
export num_points
export SHIFT_TRIES

# Define the function that will run the command
command_function() {
    local value=$1
    echo "Running ${executable} ${input_file} ${dimension} ${num_points} ${value} subset_${value}.txt"
    echo "SHIFT_TRIES=${SHIFT_TRIES:-5000} ${executable} ${input_file} ${dimension} ${num_points} ${value} subset_${value}.txt"
    SHIFT_TRIES=${SHIFT_TRIES:-5000} ${executable} ${input_file} ${dimension} ${num_points} ${value} subset_${value}.txt 2>&1 | while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line"; done > log_${value}.txt
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