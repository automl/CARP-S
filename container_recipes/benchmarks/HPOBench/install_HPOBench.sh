#!/bin/bash

CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi
$CONDA_RUN_COMMAND pip install git+https://github.com/automl/HPOBench.git@fix/numpy_deprecation
$CONDA_RUN_COMMAND pip install ConfigSpace --upgrade
$CONDA_RUN_COMMAND python container_recipes/benchmarks/HPOBench/prepare_nas_benchmarks.py