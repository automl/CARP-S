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
export CONDA_COMMAND="conda"

OPTIMIZER_CONTAINER_ID=$1
# DUMMY_Optimizer
# RandomSearch
# SMAC3
# SMAC3-1.4

BENCHMARK_ID=$2
# DUMMY_Problem
# HPOB

PYTHON_VERSION=$3
if [ -z "$PYTHON_VERSION" ]
then
      PYTHON_VERSION="3.10"
fi

# Create env
ENV_NAME="carps_${OPTIMIZER_CONTAINER_ID}_${BENCHMARK_ID}"
CREATE_COMMAND="${CONDA_COMMAND} create python=${PYTHON_VERSION} -n ${ENV_NAME} -c conda-forge -y"
echo "Creating environment:"
echo $CREATE_COMMAND
$CREATE_COMMAND

RUN_COMMAND="${CONDA_COMMAND} run -n ${ENV_NAME}"

# General
$RUN_COMMAND pip install wheel
$RUN_COMMAND pip install -r requirements.txt
$RUN_COMMAND pip install -r container_recipes/general/general_requirements_container_problem.txt
$RUN_COMMAND pip install -r container_recipes/general/general_requirements_container_optimizer.txt

# Optimizer and benchmark specific
$RUN_COMMAND pip install -r container_recipes/optimizers/${OPTIMIZER_CONTAINER_ID}/${OPTIMIZER_CONTAINER_ID}_requirements.txt
$RUN_COMMAND pip install -r container_recipes/benchmarks/${BENCHMARK_ID}/${BENCHMARK_ID}_requirements.txt

echo $(green "Done creating env! Activate with:")
echo "${CONDA_COMMAND} activate ${ENV_NAME}"
