#!/bin/bash

OPTIMIZER_CONTAINER_ID=$1

if $OPTIMIZER_CONTAINER_ID = "HEBO"
then
    git clone https://github.com/huawei-noah/HEBO.git lib/HEBOrepo
    $RUN_COMMAND pip install -e lib/HEBOrepo/HEBO
else
    $RUN_COMMAND pip install -r container_recipes/optimizers/${OPTIMIZER_CONTAINER_ID}/${OPTIMIZER_CONTAINER_ID}_requirements.txt
fi