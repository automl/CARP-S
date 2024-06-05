OPT=$1
echo $OPT

# 10.
python -m carps.run +cluster=noctua $OPT '+problem/subselection/multiobjective/dev=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MO/dev' -m 
python -m carps.run +cluster=noctua $OPT '+problem/subselection/multiobjective/test=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MO/test' -m 


