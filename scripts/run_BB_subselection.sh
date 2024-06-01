OPT=$1
echo $OPT

# Subselection. 40
python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 




# or per benchmark due to dependencies
# HPOB
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox=glob(subset_hpob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # BBOB
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox=glob(subset_bbob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # HPOBench
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox=glob(subset_hpobench*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # YAHPO
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox=glob(subset_yahpo*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 