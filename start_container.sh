#!/bin/bash

# System-dependent setup
module load system singularity
echo "${@}"

# Start the runner container - gets the hydra config and writes environment vars
# Parse whole array of args given to this script to runner.sif
echo "Starting runner container"
singularity run runner.sif
# singularity run runner.sif +optimizer/DUMMY=config +problem/DUMMY=config

# Wait for the runner container to finish
while [ ! -f "${SLURM_JOB_ID}_config.txt" ]; do
  echo "Waiting for runner container"
  sleep 1
done
echo "Runner container finished"

# Read PROBLEM_CONTAINER from file
PROBLEM_CONTAINER="$(cat "${SLURM_JOB_ID}_problem_container.txt")"
OPTIMIZER_CONTAINER="$(cat "${SLURM_JOB_ID}_optimizer_container.txt")"


# Start the problem container & wait for the flask server to start
echo "Starting problem container"
singularity instance start "${PROBLEM_CONTAINER}.sif" problem "${SLURM_JOB_ID}_config.txt"

API_URL="localhost:5000/configspace"  # Replace with the actual API URL

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ "$response" = "200" ]; then
    echo "API is up and running!"
    break
else
    echo "API is not yet ready (HTTP status: $response). Retrying in 5 seconds..."
    sleep 5
fi
done

echo "Host Found"

# Start the optimizer container
echo "Starting optimizer container"
singularity run "${OPTIMIZER_CONTAINER}.sif" "${SLURM_JOB_ID}_config.txt"

echo "All containers started"

# Remove temporary files
rm "${SLURM_JOB_ID}_config.txt"
rm "${SLURM_JOB_ID}_problem_container.txt"
rm "${SLURM_JOB_ID}_optimizer_container.txt"

# Stop the problem container
singularity instance stop problem
