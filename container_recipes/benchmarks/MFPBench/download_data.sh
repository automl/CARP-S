#!/bin/bash

CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi

if [ -z "$(ls -A carps/benchmark_data/mfpbench)" ]; then
    echo "Directory is empty, proceeding with download."
else
    echo "Directory is not empty, skipping download."
    exit 0
fi
$CONDA_RUN_COMMAND python -m mfpbench download --status --data-dir carps/benchmark_data/mfpbench
$CONDA_RUN_COMMAND python -m mfpbench download --benchmark pd1 --data-dir carps/benchmark_data/mfpbench