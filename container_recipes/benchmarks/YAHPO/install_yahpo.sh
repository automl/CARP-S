#!/bin/bash
# If you want to use yahpo locally and do not want to change to an old ConfigSpace version
# run this :)
# Run from root of repo
CONDA_ENV_NAME=$1
CARPS_ROOT=$CARPS_ROOT
PIP=$PIP

if [ -z "$PIP" ]
then
    PIP="pip"
fi
if [ -z "$CARPS_ROOT" ]
then
    CARPS_ROOT="."
fi
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi
$CONDA_RUN_COMMAND $PIP install yahpo-gym
git clone https://github.com/benjamc/yahpo_gym.git lib/yahpo_gym
$CONDA_RUN_COMMAND $PIP install -e lib/yahpo_gym/yahpo_gym
cd $CARPS_ROOT/carps
mkdir benchmark_data
cd benchmark_data
git clone https://github.com/slds-lmu/yahpo_data.git
cd ../..
$CONDA_RUN_COMMAND python $CARPS_ROOT/scripts/patch_yahpo_configspace.py
$CONDA_RUN_COMMAND $PIP install ConfigSpace --upgrade