OPT=$1
echo $OPT

# Pymoo. 10
python -m carps.run +cluster=noctua $OPT '+problem/Pymoo/MO=glob(*)' 'seed=range(1,21)' -m

# YAHPO. MO. 23
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/MO=glob(*)' 'seed=range(1,21)' -m

# MFPBench. 4
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/MO=glob(*)' 'seed=range(1,21)' -m