# SDSSP_Heuristics

This code is adapted from [this repo](https://github.com/DE0CH/SDSSP_Heuristics).
Credits go to Francois Clément and Deyao Chen. 

## Setup
### Dependencies
1. Python 3
1. gcc and associated libaries (gsl cblas), as well as common command line utilities (should be included in common linux distros)
1. [`parallel`](https://manpages.ubuntu.com/manpages/jammy/man1/parallel.1.html) tool on linux (only if you want to run in parallel)
1. moreutils

### Compile Program
If `a.out` is not there or is not working or you have changed something in the c-file: Compile shift_v2nobrute.c
```bash
gcc shift_v2nobrute.c -lm -O3
```
### On Cluster
On the cluster, make sure to have gcc and gsl cblas available.
Those commands are cluster specific. 
For the noctua2 cluster, you can load the modules like so:
```bash
ml numlib/GSL/2.7-GCC-13.2.0
ml compiler/GCC/13.2.0
```
If you are not on the cluster, install those libraries.



## Running the Subselection
This will run the subselection (i.e. choose k points the n-k remaining k points after the first iteration). 
```bash
# Create a new directory. The script will litter the directory with files and overwrite things without warning. 
```bash
mkdir run-data

# Copy the performance of optimizer per problem file into the dir
cp df_crit.csv run-data/
# $1: folder to run stuff in, $2: number of points in full set, $3: different ks
bash commands.sh run-data 106 10,20 
bash commands.sh run-data 1317 10,20 
sbatch parallel_run.sh run-data 1317

# e.g.
bash commands.sh run-data-MOMF 27 5,6,7,8,9,10,11,12,13
```

If you want to split the k points obtained from the full set of size n instead of finding k points again from
the remaining n-k, then replace `again.sh` by `split.sh` in `commands.sh` (check which arguments `again.sh`
requires).