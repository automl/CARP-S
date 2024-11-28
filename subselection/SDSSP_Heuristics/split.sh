#!/bin/bash -e

# Function to show usage
usage() {
    echo "Usage: $0 executable dimension values [-j num_cores]"
    echo "  executable   The executable to run (e.g., ./a.out)"
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

# Ensure at least 3 positional arguments are provided
if [ "$#" -lt 3 ]; then
    usage
fi

# Parse positional arguments
executable="$1"
dimension="$2"
IFS=',' read -r -a values <<< "$3"
shift 3

# Export variables needed for the command function
export executable
export dimension
export values
export reduction_factor
reduction_factor=1  # test set size = dev set size / reduction_factor

# Define the function that will run the command
command_function() {
    local value=$1
    echo "Subset size $value"
    echo "Reduction factor $reduction_factor"
    echo "Subset size after reduction $(expr ${value} / ${reduction_factor}"
    # make a temp file
    tpfile=$(mktemp)
    python3 ../extract_csv.py subset_${value}.csv ${tpfile}
    SHIFT_TRIES=5000 ${executable} ${tpfile} ${dimension} ${value} $(expr ${value} / ${reduction_factor}) half_subset_${value}.txt 2>&1 | ts > log_half_subset_${value}.txt
    python3 ../match_index.py subset_${value}.csv half_subset_${value}.txt half_subset_${value}.csv subset_complement_subset_${value}.csv
    python3 ../extract_csv.py subset_complement_subset_${value}.csv subset_complement_subset_${value}.txt
    SHIFT_TRIES=1 ${executable} subset_complement_subset_${value}.txt ${dimension} ${value} $(expr ${value} / ${reduction_factor}) subset_complement_subset_${value}.txt 2> /dev/null > /dev/null
    head -n 1 half_subset_${value}.txt | python3 -c 'import sys; import re; print("half subset:", f"k={sys.argv[1]}", re.search(r"discrepancy=([\d\.]+)", input())[0])' "${value}"
    head -n 1 subset_complement_subset_${value}.txt | python3 -c 'import sys; import re; print("other half subset:", f"k={sys.argv[1]}", re.search(r"discrepancy=([\d\.]+)", input())[0])' "${value}"
    rm $tpfile
}

# Export the function if parallel execution is used
if [ "$num_cores" -gt 1 ]; then
    export -f command_function
    # Run the commands in parallel with the specified number of cores
    parallel -j "$num_cores" command_function ::: "${values[@]}"
else
    # Run the commands sequentially
    for value in "${values[@]}"; do
        echo $value
        command_function "$value"
    done
fi