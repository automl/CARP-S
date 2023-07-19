#!/bin/bash

# System-dependent setup
module load system singularity

# Start the runner container - gets the hydra config and writes environment vars
# Parse whole array of args given to this script to runner.sif
echo "Starting runner container"
singularity run runner.sif "${SLURM_JOB_ID}" "${@}"

# Wait for the runner container to finish
while [ ! -f "${SLURM_JOB_ID}_config.yaml" ]; do
  echo "Waiting for runner container"
  sleep 1
done
echo "Runner container finished"

# Read PROBLEM_CONTAINER from file
PROBLEM_CONTAINER=$(cat "${SLURM_JOB_ID}_problem_container.txt")
OPTIMIZER_CONTAINER=$(cat "${SLURM_JOB_ID}_optimizer_container.txt")

# Start the problem container & wait for the flask server to start
echo "Starting problem container"
singularity run "${PROBLEM_CONTAINER}.sif" "${SLURM_JOB_ID}_config.yaml"

while ! ping -c1 localhost:5000 &>/dev/null; do
  echo "Waiting for Server"
  sleep
done

echo "Host Found"


# Start the optimizer container
echo "Starting optimizer container"
singularity run "${OPTIMIZER_CONTAINER}.sif" "${SLURM_JOB_ID}_config.yaml"

echo "All containers started"
