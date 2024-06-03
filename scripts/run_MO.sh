OPT=$1
echo $OPT

# Pymoo. 10
python -m carps.run +cluster=noctua $OPT '+problem/Pymoo/MO=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# YAHPO. MO. 23
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/MO=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# MFPBench. 4
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/MO=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# HPOBench. 88
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/MO/tab=glob(*)' 'seed=range(1,11)' baserundir=runs_MO -m
python -m carps.run +cluster=noctua $OPT '+problem/HPOBench/MO/tab=glob(*)' 'seed=range(11,21)' baserundir=runs_MO -m
