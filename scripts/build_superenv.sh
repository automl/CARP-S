#!/bin/bash

ml lang/Anaconda3/2022.05


# Color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

function red {
    printf "${RED}$@${NC}\n"
}

function green {
    printf "${GREEN}$@${NC}\n"
}

function yellow {
    printf "${YELLOW}$@${NC}\n"
}


export CONT_GENERAL_PATH=containers/general
export CONT_GENERAL_RECIPE_PATH=container_recipes/general
export CONT_BENCH_PATH=containers/benchmarks
export CONT_BENCH_RECIPE_PATH=container_recipes/benchmarks
export CONT_OPT_PATH=containers/optimizers
export CONT_OPT_RECIPE_PATH=container_recipes/optimizers

PYTHON_VERSION=$2
# if [ -z "$PYTHON_VERSION" ]
# then
#       PYTHON_VERSION="3.10"
# fi

# ENV_LOCATION=$4

CONDA_COMMAND=$3
if [ -z "$CONDA_COMMAND" ]
then
    CONDA_COMMAND="conda"
fi

EXTRA_COMMAND=$4


# Create env
# ENV_NAME="carps_${OPTIMIZER_CONTAINER_ID}"  #carps_${OPTIMIZER_CONTAINER_ID}_${BENCHMARK_ID}"
ENV_NAME=$1
if [ -z "$ENV_LOCATION" ]
then
    ENV_LOCATION="-n ${ENV_NAME}"
else
    ENV_LOCATION="-p ${ENV_LOCATION}/${ENV_NAME}"
fi

find_in_conda_env(){
    $CONDA_COMMAND env list | grep "${@}" >/dev/null 2>/dev/null
}
CREATE_COMMAND="${CONDA_COMMAND} create python=${PYTHON_VERSION} -c conda-forge ${ENV_LOCATION} -y"


if find_in_conda_env ".*${ENV_NAME}.*" ; then
   echo "Env already exists"
else 
    echo "Creating environment:"
    echo $CREATE_COMMAND
    $CREATE_COMMAND
fi

RUN_COMMAND="${CONDA_COMMAND} run ${ENV_LOCATION}"

# General
$RUN_COMMAND pip install wheel
$RUN_COMMAND pip install swig
$RUN_COMMAND pip install -e .
$RUN_COMMAND pip install -r requirements.txt
$RUN_COMMAND pip install -r container_recipes/general/general_requirements_container_problem.txt
$RUN_COMMAND pip install -r container_recipes/general/general_requirements_container_optimizer.txt

# Benchmark specific
for benchmark_id in "HPOB" "BBOB" "MFPBench" "Pymoo"
do
    $RUN_COMMAND pip install -r container_recipes/benchmarks/${benchmark_id}/${benchmark_id}_requirements.txt
done

# YAHPO
bash container_recipes/benchmarks/YAHPO/prepare_yahpo.sh $ENV_LOCATION

# MFPBench
bash container_recipes/benchmarks/MFPBench/download_data.sh $ENV_LOCATION

# HPOBench
# $RUN_COMMAND pip install git+https://github.com/automl/HPOBench.git
git clone https://github.com/automl/HPOBench.git lib/HPOBench
# RUN_COMMAND=
$RUN_COMMAND pip install Cython==0.29.36
$RUN_COMMAND pip install scikit-learn==0.24.2 --no-build-isolation
$RUN_COMMAND pip install openml==0.12.2
$RUN_COMMAND pip install xgboost==1.3.1
$RUN_COMMAND pip install lib/HPOBench
# Cheat a bit, works with newer ConfigSpace
# $RUN_COMMAND pip install ConfigSpace==0.6.1

$RUN_COMMAND $EXTRA_COMMAND

echo $(green "Done creating env! Activate with:")
echo "${CONDA_COMMAND} activate ${ENV_NAME}"
