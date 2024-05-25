OPT=$1
echo $OPT
# MFPBench PD1. 4
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/pd1=glob(*)' 'seed=range(1,21)' -m

# MFPBench mfh. 8
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/mfh=glob(*)' 'seed=range(1,21)' -m