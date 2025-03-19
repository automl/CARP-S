#!/bin/bash

CONDA_ENV_NAME=$1
PIP=$PIP

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
$CONDA_RUN_COMMAND $PIP install git+https://github.com/automl/HPOBench.git@fix/numpy_deprecation
$CONDA_RUN_COMMAND $PIP install openml
$CONDA_RUN_COMMAND $PIP install ConfigSpace --upgrade
$CONDA_RUN_COMMAND python container_recipes/benchmarks/HPOBench/prepare_nas_benchmarks.py

# Build container
# This is necessary because the specific container does not get built off the correct
# branch in HPOBench, thus the container in the registry does not work.
HPOBENCH_CONTAINER_DIR=$(python container_recipes/benchmarks/HPOBench/get_container_dir.py)
echo "Building HPOBench container in $HPOBENCH_CONTAINER_DIR"
apptainer build $HPOBENCH_CONTAINER_DIR/ml_mmfb_0.0.1 lib/HPOBench/hpobench/container/recipes/ml/Singularity.ml_mmfb
echo "Built $HPOBENCH_CONTAINER_DIR/ml_mmfb_0.0.1"

apptainer build $HPOBENCH_CONTAINER_DIR/nasbench_201_0.0.5 lib/HPOBench/hpobench/container/recipes/nas/Singularity.nasbench_201 
echo "Built $HPOBENCH_CONTAINER_DIR/nasbench_201_0.0.5"