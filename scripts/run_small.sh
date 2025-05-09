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
YAHPO_SO='+task/YAHPO/SO=glob("cfg_*")'
BBOB_2D='+task/BBOB=glob("cfg_2_*")'

python -m carps.run $SMAC20 $YAHPO_SO $SEED --multirun
# python -m carps.run $SMAC14 $YAHPO_SO $SEED --multirun
python -m carps.run $RANDOMSEARCH $YAHPO_SO $SEED --multirun
python -m carps.run $HEBO $YAHPO_SO $SEED --multirun

python -m carps.run $SMAC20 $BBOB_2D $SEED --multirun
# python -m carps.run $SMAC14 $BBOB_2D $SEED --multirun
python -m carps.run $RANDOMSEARCH $BBOB_2D $SEED --multirun
python -m carps.run $HEBO $BBOB_2D $SEED --multirun

# ##############################################
# # Multi-objective blackbox
# YAHPO_MO='+task/YAHPO/MO=glob("cfg_*")'

# python -m carps.run $SMAC20 $YAHPO_MO $SEED --multirun
# python -m carps.run $SMAC14 $YAHPO_MO $SEED --multirun
# python -m carps.run $RANDOMSEARCH $YAHPO_MO $SEED --multirun
# python -m carps.run $HEBO $YAHPO_MO $SEED --multirun


# ##############################################
# # Multi-fidelity
# WALLTIME_LIMIT=0  # TODO tbd