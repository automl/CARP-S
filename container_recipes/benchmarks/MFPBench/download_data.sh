#!/bin/bash

CONDA_ENV_NAME=$1
if [ -z "$CONDA_ENV_NAME" ]
then
    CONDA_RUN_COMMAND=
else
    CONDA_RUN_COMMAND="${CONDA_COMMAND} run ${CONDA_ENV_NAME}"
fi

$CONDA_RUN_COMMAND python -m mfpbench download --status --data-dir data
$CONDA_RUN_COMMAND python -m mfpbench download --benchmark pd1