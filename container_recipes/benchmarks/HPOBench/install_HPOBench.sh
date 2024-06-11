#!/bin/bash

CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi
$CONDA_RUN_COMMAND pip install git+https://github.com/automl/HPOBench.git --ignore-requires-python
$CONDA_RUN_COMMAND pip install tqdm
$CONDA_RUN_COMMAND pip install pandas==1.2.4
$CONDA_RUN_COMMAND pip install Cython==0.29.36
$CONDA_RUN_COMMAND pip install scikit-learn==0.24.2 --no-build-isolation  # <- no buil isolation is important
$CONDA_RUN_COMMAND pip install openml==0.12.2
$CONDA_RUN_COMMAND pip install xgboost==1.3.1
$CONDA_RUN_COMMAND pip install ConfigSpace #==0.6.1