Blackbox problems

1. YAHPO gym Set Single-Objective
```bash
SMAC20='+optimizer/smac20=blackbox'
SMAC14='+optimizer/smac14=blackbox'
RANDOMSEARCH='+optimizer/randomsearch=config'
HEBO='+optimizer/hebo=config'

YAHPO_SO='+problem/YAHPO/SO=glob("cfg_*")'

SEED='seed=range(1,11)'


python carps/container/create_cluster_configs.py $SMAC20 $YAHPO_SO $SEED --multirun
python carps/container/create_cluster_configs.py $SMAC14 $YAHPO_SO $SEED --multirun
python carps/container/create_cluster_configs.py $OPTIMIZER $YAHPO_SO $SEED --multirun
python carps/container/create_cluster_configs.py $HEBO $YAHPO_SO $SEED --multirun

```


MF


fix/test multi-fidelity
overview experiments
create initial experiment set


PEACH
FISH:  framework for integrated scientific harvesting
CARP: comprehensive automl research platform