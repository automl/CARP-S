#!/bin/bash

# ----- System-dependent setup - Adapt to your system ----- #
module load system singularity
# the BENCHMARKING_JOB_ID has to be set to a unique id for each run
export BENCHMARKING_JOB_ID=$SLURM_JOB_ID

# for script cleanup
cleanup_files() {
  echo "Cleaning up temporary files and containers"
  rm "${BENCHMARKING_JOB_ID}_hydra_config.yaml"
  rm "${BENCHMARKING_JOB_ID}_pyexperimenter_id.txt"
  rm "${BENCHMARKING_JOB_ID}_problem_container.txt"
  rm "${BENCHMARKING_JOB_ID}_optimizer_container.txt"

  singularity instance stop problem
}

trap "cleanup_files" EXIT

# Start the runner container - gets the hydra config and writes environment vars
echo "Starting runner container"
singularity run containers/general/runner.sif

# Wait for the runner container to finish
while [ ! -f "${BENCHMARKING_JOB_ID}_pyexperimenter_id.txt" ]; do
  echo "Waiting for runner container"
  sleep 1
done
echo "Runner container finished"

# Read PROBLEM_CONTAINER from file
PROBLEM_CONTAINER="$(cat "${BENCHMARKING_JOB_ID}_problem_container.txt")"
OPTIMIZER_CONTAINER="$(cat "${BENCHMARKING_JOB_ID}_optimizer_container.txt")"


# Start the problem container & wait for the flask server to start
echo "Starting problem container"
singularity instance start "containers/benchmarks/${PROBLEM_CONTAINER}.sif" problem

API_URL="localhost:5000/configspace"

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ "$response" = "200" ]; then
    echo "API is up and running!"
    break
else
    # get the path to the error files
    LOG_ERR_PATH=$(singularity instance list -j |
     python -c "import sys, json;\
     output = json.load(sys.stdin)['instances'];\
     print(next(item['logErrPath'] for item in output if item.get('instance', None) == 'problem'))"
    )

    printf "#%.0s" {1..100}; echo
    echo "Run fails because"
    tail -n 100  $LOG_ERR_PATH
    printf "#%.0s" {1..100}; echo
    echo "For further information, please refer to the error logs under $LOG_ERR_PATH"

    echo "API is not yet ready (HTTP status: $response). Retrying in 5 seconds..."
    sleep 5
fi
done

echo "Host Found"

# Start the optimizer container
echo "Starting optimizer container"
singularity exec "containers/optimizers/${OPTIMIZER_CONTAINER}.sif" python carps/container/container_script_optimizer.py

echo "Run Finished"
