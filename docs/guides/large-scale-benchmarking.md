# Large Scale Benchmarking

## Parallel
You can run your optimization via
```bash
conda run -n automlsuite_DUMMY_Optimizer_DUMMY_ObjectiveFunction python carps/run.py \
    +optimizer/DUMMY=config +problem/DUMMY=config \
    'seed=range(1,11)' \
    +cluster=local -m
```
This uses joblib parallelization on your local machine.
If you are on a slurm cluster, you can specify `+cluster=slurm` and adapt this to your needs.
Check [this page for more launchers](https://hydra.cc/docs/plugins/joblib_launcher/), e.g. Ray or RQ besides Joblib and Submitit.
