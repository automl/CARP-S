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
If `a.out` is not there or you have changed something in the c-file: Compile shift_v2nobrute.c
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

# Single k
python subselect.py k=30

# Sweep over ks
python subselect.py k=5,10,15,20 -m
```
Check `config.yaml` or `subselect.py` for more arguments.

## Inspecting the subselection
See `show_results_info.ipynb`.