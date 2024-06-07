# SDSSP_Heuristics

This code is adapted from [this repo](https://github.com/DE0CH/SDSSP_Heuristics).
Credits go to Francois Clément and Deyao Chen. 

## Dependencies:
1. Python 3
1. gcc and associated libaries, as well as common command line utilities (should be included in common linux distros)
1. [`parallel`](https://manpages.ubuntu.com/manpages/jammy/man1/parallel.1.html) tool on linux (only if you want to run in parallel)
1. moreutils


## Steps to reproduce:
1. Compile shift_v2nobrute.c
```bash
gcc shift_v2nobrute.c -lm -O3
```
1. Make a directory. The script will litter the directory with files and overwrite things without warning. 
```bash
mkdir run-data
```
1. cd to the directory. WARNING: the scripts will overwrite files in cwd without warning.
```bash
cd run-data
```
1. put your df_crit.csv in the directory and run (for example). See the help of each files for usage.
```bash
python3 ../extract_csv.py df_crit.csv df_crit.txt
../run.sh -j 10 ../a.out df_crit.txt 3 2188 30,60,80,90,100,110,120
../format.sh df_crit.csv 30,60,80,90,100,110,120
../split.sh -j 10 ../a.out 3 30,60,80,90,100,110,120
```


If you want to do the other way of splitting (i.e. choose k points the n-k remaining k points after the first iteration). 

```
python3 ../extract_csv.py df_crit.csv df_crit.txt
../run.sh -j 10 ../a.out df_crit.txt 3 2188 30,60,80,90,100,110,120
../format.sh df_crit.csv 30,60,80,90,100,110,120 
../again.sh -j 9 ../a.out df_crit.csv 1857 30,60,80,90,100,110,120 
```

1. print all the discrepancies into a csv file
```bash
regex='^.*k=([0-9]+).*discrepancy=(0\.[0-9]+).*$'
echo 'which,k,discrepancy'
for value in 30 60 80 90 100 110 120
do
    if [[ $(head -n 1 subset_$value.txt) =~ $regex ]]
    then 
        echo "s1,${BASH_REMATCH[1]},${BASH_REMATCH[2]}"
    fi
    if [[ $(head -n 1 subset_complement_subset_$value.txt) =~ $regex ]]
    then
        echo "s2,${BASH_REMATCH[1]},${BASH_REMATCH[2]}"
    fi
done
```


## Alternative:
```bash
mkdir run-data
cp df_crit.csv run-data/
# $1: folder to run stuff in, $2: number of points in full set, $3: different ks
bash commands.sh run-data 106 10,20 



bash commands.sh run-data-MOMF 27 5,6,7,8,9,10,11,12,13
```



