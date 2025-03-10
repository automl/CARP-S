OPT=$1
echo $OPT

# HPOBench surr. 7
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/blackbox/surr=glob(*)' 'seed=range(1,21)' -m 

# HPOBench tab ML. 88
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/blackbox/tab/ml=glob(*)' 'seed=range(1,11)' -m
python -m carps.run +cluster=noctua $OPTs '+task/HPOBench/blackbox/tab/ml=glob(*)' 'seed=range(11,21)' -m

# HPOBench tab nas. 10
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/blackbox/tab/nas=glob(*)' 'seed=range(1,21)' -m