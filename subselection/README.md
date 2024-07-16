# Subselection

## Subset Visualization
For the plots in the paper, download the [data](https://drive.google.com/file/d/17GtnHx6MKFSB4f89G1MLYHJi-ZHnNFMT/view?usp=sharing) and unzip the folder in `subselection`.
The folder structure should be e.g. `subselection/data/MOMF/lognorm`.
Then run the notebook `inspect.ipynb`.

## Process
1. Run full set of experiments with three optimizers. The optimizers should have different behaviors.
1. Gather the data for the optimizers with `python -m carps.analysis.gather_data rundir`. In the case of multi-objective (or multi-objective-multi-fidelity), do `python -m carps.analysis.gather_data rundir trajectory_logs.jsonl`.
1. Run `python get_fullset.py runs_MOMF '["RandomSearch","SMAC3-MOMF-GP","Nevergrad-DE"]' subselection/data/MOMF/default`. This converts the semi-raw logs to the format required for the subselection. Creates `df_crit.csv` at `subselection/data/MOMF/default`. Contains the normalized performance data averaged per seed for each `problem_id` (rows) and `optimizer_id` (cols).
1. Check the performance space with `subselection/inspect.ipynb`. If it looks clumped, use the generated `df_crit_log_norm.csv` (log transformed and normalized again) for the next step instead.
1. Copy the `df_crit.csv` into the SDSS_Heuristics folder and check there how to subselect. Points must be in [0,1]. Run the subselection. See the `subselection/SDSSP_Heuristics/README.md` for details.
1. Copy the results back into `subselection/data/MOMF/default`, adjust the paths in `subselection/inspect.ipynb` and run `subselection/inspect.ipynb` to see results and analyse the subselection. Remember the subset size k `subselection/inspect.ipynb` proposes to use. If you ran on the log norm data, copy into `subselection/data/MOMF/lognorm/`.
1. Create the subset configs with e.g. `python subselection/create_subset_configs.py subselection/data/MOMF/lognorm/subset_9.csv subselection/data/MOMF/lognorm/subset_complement_subset_9.csv momf`.
1. Run everything again on the subset 🙂, e.g. create a bash script and run `bash script/run_MOMF_subselection.sh '+optimizer/smac20=momf_gp'`. Script can look like this:
```bash
OPT=$1
echo $OPT

# 9 each = 2* [9 * 20 jobs] = 2 * 180 jobs
python -m carps.run +cluster=noctua $OPT '+problem/subselection/momf/dev=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MOMF/dev' -m 
python -m carps.run +cluster=noctua $OPT '+problem/subselection/momf/test=glob(*)' 'seed=range(1,21)' 'baserundir=runs_subset_MOMF/test' -m 
```