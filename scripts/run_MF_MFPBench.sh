OPT=$1
echo $OPT
# MFPBench PD1. 4
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/SO/pd1=glob(*)' 'seed=range(1,21)'  'hydra.launcher.mem_per_cpu=32G' -m

# MFPBench mfh. 8
python -m carps.run +cluster=noctua $OPT '+problem/MFPBench/SO/mfh=glob(*)' 'seed=range(1,21)' -m