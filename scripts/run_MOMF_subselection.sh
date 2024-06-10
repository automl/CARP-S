OPT=$1
echo $OPT

# 9 each = 2* [9 * 20 jobs] = 2 * 180 jobs
python -m carps.run +cluster=noctua $OPT '+problem/subselection/momf/dev=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MOMF/dev' -m 
python -m carps.run +cluster=noctua $OPT '+problem/subselection/momf/test=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MOMF/test' -m 


