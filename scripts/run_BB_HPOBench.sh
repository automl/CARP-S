OPT=$1
echo $OPT

# HPOBench surr. 7
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/blackbox/surr=glob(*)' 'seed=range(1,21)' -m 

# HPOBench tab. 88
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/blackbox/tab=glob(*)' 'seed=range(1,11)' -m
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/blackbox/tab=glob(*)' 'seed=range(11,21)' -m