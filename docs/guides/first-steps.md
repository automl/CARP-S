# First Steps

First, follow the [installation instructions](../installation.md) to install CARP-S. As described
in the installation guide, make sure to install the requirements for the benchmark and optimizer
you would like to run, e.g. `SMAC2.0` and `BBOB`.

Once the requirements for both an optimizer and a benchmark are installed, you can run one of 
the following minimal examples to benchmark `SMAC2.0` on `BBOB` directly with Hydra:

```bash
# Run SMAC BlackBoxFacade on certain BBOB problem
python -m carps.run +optimizer/smac20=blackbox +problem/BBOB=cfg_4_1_4_0 seed=1 task.n_trials=25

# Run SMAC BlackBoxFacade on all available BBOB problems for 10 seeds
python -m carps.run +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,11)' -m
```

For the second command, the Hydra -m (or --multirun) option indicates that multiple runs will be 
performed over a range of parameter values. In this case, it's indicating that the benchmarking
should be run for all available BBOB problems (+problem/BBOB=glob(*)) and for 10 different 
seed values (seed=range(1,11)).