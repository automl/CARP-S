#!/bin/bash
# If you want to use yahpo locally and do not want to change to an old ConfigSpace version
# run this :)
# Run from root of repo
CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi
$CONDA_RUN_COMMAND pip install yahpo-gym
cd carps
mkdir benchmark_data
cd benchmark_data
git clone https://github.com/slds-lmu/yahpo_data.git
cd ../..
$CONDA_RUN_COMMAND python scripts/patch_yahpo_configspace.py
$CONDA_RUN_COMMAND pip install ConfigSpace --upgrade