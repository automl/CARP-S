#!/bin/bash
#SBATCH --job-name=cont-db
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=1-300
#SBATCH --output=slurmlogs/carps-cont-db-%A_%a.out


# Prerequisites:
# Build the container image with `apptainer build container/env.sif container/env.def`

# Load Apptainer (if not already loaded)
module load tools Apptainer

# Set the container image location
CONTAINER_PATH=container/env.sif

# Command to run Hydra multirun using the container's Python environment

for i in $(seq 1 12); do
    echo "Running job number $i"
    # Run the command inside the container
    apptainer exec $CONTAINER_PATH python -m carps.run_from_db job_nr_dummy=%a
done