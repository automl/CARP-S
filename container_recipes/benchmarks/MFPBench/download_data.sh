#!/bin/bash

CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi
DATA_PATH=$(python -c "from carps.objective_functions.mfpbench import MFPBENCH_TASK_DATA_DIR; print(MFPBENCH_TASK_DATA_DIR)")

if [ -z "$(ls -A $DATA_PATH)" ]; then
    echo "Directory is empty, proceeding with download."
else
    echo "Directory $DATA_PATH is not empty, skipping download."
    exit 0
fi
$CONDA_RUN_COMMAND python -m mfpbench download --status --data-dir $DATA_PATH
$CONDA_RUN_COMMAND python -m mfpbench download --benchmark pd1 --data-dir $DATA_PATH