OPT=$1
echo $OPT

# Subselection. 30
python -m carps.run +cluster=noctua $OPT '+problem/subselection/multifidelity/dev=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MF/dev' -m 
python -m carps.run +cluster=noctua $OPT '+problem/subselection/multifidelity/test=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MF/test' -m 




