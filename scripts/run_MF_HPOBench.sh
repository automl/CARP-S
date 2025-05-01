OPT=$1
echo $OPT
# HPOBench MF. 156
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/multifidelity=glob(*)' 'seed=range(1,7)' hydra.launcher.timeout_min=720 -m
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/multifidelity=glob(*)' 'seed=range(7,13)' hydra.launcher.timeout_min=720 -m
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/multifidelity=glob(*)' 'seed=range(13,19)' hydra.launcher.timeout_min=720 -m
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/multifidelity=glob(*)' 'seed=range(19,21)' hydra.launcher.timeout_min=720 -m