OPT=$1
echo $OPT

# Subselection. 30
python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/dev=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_BB/dev' -m 
python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/test=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_BB/test' -m 




# or per benchmark due to dependencies
# HPOB
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/dev=glob(subset_hpob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/test=glob(subset_hpob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # BBOB
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/dev=glob(subset_bbob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/test=glob(subset_bbob*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # HPOBench
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/dev=glob(subset_hpobench*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/test=glob(subset_hpobench*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# # YAHPO
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/dev=glob(subset_yahpo*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 
# python -m carps.run +cluster=noctua $OPT '+problem/subselection/blackbox/test=glob(subset_yahpo*)' 'seed=range(1,21)' 'baserundir=runs_subset' -m 