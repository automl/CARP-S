OPT=$1
echo $OPT
# HPOBench MF. 156
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/multifidelity=glob(*)' 'seed=range(1,7)' -m
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/multifidelity=glob(*)' 'seed=range(7,13)' -m
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/multifidelity=glob(*)' 'seed=range(13,19)' -m
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/multifidelity=glob(*)' 'seed=range(19,21)' -m