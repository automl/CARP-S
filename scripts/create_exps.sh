#!/bin/bash
#SBATCH --job-name=createexp            # Job name
#SBATCH --output=createexp.txt   # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (usually 1 for single jobs)
#SBATCH --cpus-per-task=32           # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=8G                     # Memory per node (adjust as needed)
#SBATCH --time=24:00:00              # Max run time (format: HH:MM:SS)
#SBATCH --partition=normal         # Partition to submit to (use the appropriate one for your system)
#SBATCH --mail-type=END,FAIL         # Email notifications for job start, end, or failure
#SBATCH --mail-user=c.benjamins@ai.uni-hannover.de  # Your email address

module load lang/Miniforge3/24.1.2-0
conda init
source ~/.bashrc
conda activate carpsexp
bash scripts/create_experiments_in_db.sh