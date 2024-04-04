SEED='seed=range(1,11)'

##############################################
# Optimizers 
SMAC20='+optimizer/smac20=blackbox'
SMAC14='+optimizer/smac14=blackbox'
RANDOMSEARCH='+optimizer/randomsearch=config'
HEBO='+optimizer/hebo=config'
STBO='+optimizer/synetune=BO'

# MF
SMAC20MF='+optimizer/smac20=multifidelity'
STBOHB='+optimizer/synetune=BOHB'
STDEHB='+optimizer/synetune=DEHB'
STASHA='+optimizer/synetune=ASHA'

##############################################
# Blackbox 
YAHPO_SO='+problem/YAHPO/SO=glob("cfg_*")'

python smacbenchmarking/container/create_cluster_configs.py $SMAC20 $YAHPO_SO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $SMAC14 $YAHPO_SO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $OPTIMIZER $YAHPO_SO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $HEBO $YAHPO_SO $SEED --multirun

##############################################
# Multi-objective blackbox
YAHPO_MO='+problem/YAHPO/MO=glob("cfg_*")'

python smacbenchmarking/container/create_cluster_configs.py $SMAC20 $YAHPO_MO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $SMAC14 $YAHPO_MO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $OPTIMIZER $YAHPO_MO $SEED --multirun
python smacbenchmarking/container/create_cluster_configs.py $HEBO $YAHPO_MO $SEED --multirun


##############################################
# Multi-fidelity
WALLTIME_LIMIT=0  # TODO tbd