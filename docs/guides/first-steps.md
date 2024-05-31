# First Steps

Once the requirements for both an optimizer and a benchmark, e.g. `SMAC2.0` and `BBOB`, 
are installed, you can run one of the following minimal examples to benchmark 
`SMAC2.0` on `BBOB` directly with Hydra:

```bash
# Run SMAC BlackBoxFacade on certain BBOB problem
python -m carps.run +optimizer/smac20=blackbox +problem/BBOB=cfg_4_1_4_0 seed=1 task.n_trials=25

# Run SMAC BlackBoxFacade on all available BBOB problems for 10 seeds
python -m carps.run +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,11)' -m
```