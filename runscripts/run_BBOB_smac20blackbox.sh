#!/bin/bash

#SBATCH --job-name=smac20_BBOB
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
#SBATCH --mail-user=c.benjamins@ai.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --array=2-4
#SBATCH --output slurm_logs/smac20_BBOB-job_%A_%a.out
#SBATCH --error slurm_logs/smac20_BBOB-job_%A_%a.err

# Move to location of the runscript

# Activate the conda env
micromamba activate /scratch/hpc-prf-intexml/cbenjamins/envs/expl2

# Create run configs
# python smacbenchmarking/container/create_cluster_configs.py +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,21)' -m

./start_container_noctua2.sh
