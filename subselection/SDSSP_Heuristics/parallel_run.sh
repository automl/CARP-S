#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH -J "subselect"
#SBATCH -p normal
#SBATCH --array=10,20,30,40,50,60,70,80,90,100,150,200,250,500
#SBATCH --mem-per-cpu 8G
#SBATCH -o slurm/slurm-%j.out

# ,30,40,50,60,70,80,90,100,150,200,250,500

folder=$1 # folder to run stuff in
n_tasks=$2  # number of points
#ks=$3  # subset sizes as comma separated list

echo "Folder: $1"
echo "n_tasks: $2"

echo ">>>1 current working dir"
pwd

cd $folder
echo ">>>2 current working dir"
pwd
rm subset*
rm log*
rm info.csv
rm df_crit.txt 
cd -
echo ">>>3 current working dir"
pwd

bash commands.sh $folder $n_tasks $SLURM_ARRAY_TASK_ID, clean