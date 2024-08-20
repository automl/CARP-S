Run with 

```bash
# hangs
python -m carps.run +optimizer/smac20=blackbox +problem/BBOB=cfg_2_1_2_0 task.n_workers=4

# API needs to be adjusted
python -m carps.run +optimizer/optuna=blackbox +problem/BBOB=cfg_2_1_2_0 task.n_workers=4

# works
python -m carps.run +optimizer/randomsearch=config +problem/BBOB=cfg_2_1_2_0 task.n_workers=4
```