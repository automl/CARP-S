#!/bin/bash
# If you want to use yahpo locally and do not want to change to an old ConfigSpace version
# run this :)
# Run from root of repo
CONDA_ENV_NAME=$1
PIP=$PIP
CARPS_ROOT=$(python -c "from carps.utils.env_vars import CARPS_ROOT; print(CARPS_ROOT)")

if [ -z "$PIP" ]
then
    PIP="pip"
fi

if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi

# Install yahpo-gym
git clone https://github.com/automl/yahpo_gym.git lib/yahpo_gym
$CONDA_RUN_COMMAND $PIP install -e lib/yahpo_gym/yahpo_gym

# Get task data
YAHPO_TASK_DATA_DIR=$(python -c "from carps.objective_functions.yahpo import YAHPO_TASK_DATA_DIR; print(YAHPO_TASK_DATA_DIR)")
mkdir -p $YAHPO_TASK_DATA_DIR
git clone https://github.com/slds-lmu/yahpo_data.git $YAHPO_TASK_DATA_DIR
$CONDA_RUN_COMMAND python $CARPS_ROOT/container_recipes/benchmarks/YAHPO/patch_yahpo_configspace.py
$CONDA_RUN_COMMAND $PIP install ConfigSpace --upgrade