OPT=$1
echo $OPT

# YAHPO. MO. 35
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/MOMF=glob(*)' 'seed=range(1,21)' baserundir=runs_MOMF -m

