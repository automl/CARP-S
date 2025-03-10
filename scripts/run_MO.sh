OPT=$1
echo $OPT

# Pymoo. 10 + 20
python -m carps.run +cluster=noctua $OPT '+task/Pymoo/MO/unconstraint=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m
python -m carps.run +cluster=noctua $OPT '+task/Pymoo/ManyO/unconstraint=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# YAHPO. MO. 23
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/MO=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# MFPBench. 4
python -m carps.run +cluster=noctua $OPT '+task/MFPBench/MO=glob(*)' 'seed=range(1,21)' baserundir=runs_MO -m

# HPOBench. 88
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/MO/tab=glob(*)' 'seed=range(1,11)' baserundir=runs_MO -m
python -m carps.run +cluster=noctua $OPT '+task/HPOBench/MO/tab=glob(*)' 'seed=range(11,21)' baserundir=runs_MO -m
