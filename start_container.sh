#!/bin/bash

# System-dependent setup
module load system singularity
echo "${@}"

# Start the runner container - gets the hydra config and writes environment vars
# Parse whole array of args given to this script to runner.sif
echo "Starting runner container"
singularity run runner.sif "${@}"
# singularity run runner.sif +optimizer/DUMMY=config +problem/DUMMY=config

# Wait for the runner container to finish
while [ ! -f "${SLURM_JOB_ID}_config.txt" ]; do
  echo "Waiting for runner container"
  sleep 1
done
echo "Runner container finished"

# Read PROBLEM_CONTAINER from file
export PROBLEM_CONTAINER="$(cat "${SLURM_JOB_ID}_problem_container.txt")"
export OPTIMIZER_CONTAINER="$(cat "${SLURM_JOB_ID}_optimizer_container.txt")"

cat "${SLURM_JOB_ID}_config.txt"

# Start the problem container & wait for the flask server to start
echo "Starting problem container"
singularity run "${PROBLEM_CONTAINER}.sif" "${SLURM_JOB_ID}_config.txt"

while ! ping -c1 localhost:5000 &>/dev/null; do
  echo "Waiting for Server"
  sleep 1
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
