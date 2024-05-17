#!/bin/bash
# bash scripts/build_env.sh OPTIMIZER_CONTAINER_ID BENCHMARK_ID PYTHON_VERSION 
export CONT_GENERAL_PATH=containers/general
export CONT_GENERAL_RECIPE_PATH=container_recipes/general
export CONT_BENCH_PATH=containers/benchmarks
export CONT_BENCH_RECIPE_PATH=container_recipes/benchmarks
export CONT_OPT_PATH=containers/optimizers
export CONT_OPT_RECIPE_PATH=container_recipes/optimizers
export ENV_LOCATION="/scratch/hpc-prf-intexml/carps-neurips24/envs"

# bash ${CONT_BENCH_RECIPE_PATH}/HPOB/download_data.sh
# bash scripts/prepare_yahpo.sh

bash scripts/build_env.sh HPOB SMAC3-1.4 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOB RandomSearch 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOB HEBO 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOB SyneTune 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOB DEHB 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO SMAC3 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO SMAC3-1.4 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO RandomSearch 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO HEBO 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO SyneTune 3.11 $ENV_LOCATION
bash scripts/build_env.sh YAHPO DEHB 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench SMAC3 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench SMAC3-1.4 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench RandomSearch 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench HEBO 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench SyneTune 3.11 $ENV_LOCATION
bash scripts/build_env.sh HPOBench DEHB 3.9 $ENV_LOCATION
bash scripts/build_env.sh BBOB SMAC3 3.11 $ENV_LOCATION
bash scripts/build_env.sh BBOB SMAC3-1.4 3.11 $ENV_LOCATION
bash scripts/build_env.sh BBOB RandomSearch 3.11 $ENV_LOCATION
bash scripts/build_env.sh BBOB HEBO 3.11 $ENV_LOCATION
bash scripts/build_env.sh BBOB SyneTune 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench SMAC3 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench SMAC3-1.4 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench RandomSearch 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench HEBO 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench SyneTune 3.11 $ENV_LOCATION
bash scripts/build_env.sh MFPBench DEHB 3.11 $ENV_LOCATION

# for benchmark in "HPOB" "YAHPO" "HPOBench" "BBOB" "MFPBench"
# do
#     for optimizer in "SMAC3" "SMAC3-1.4" "RandomSearch" "HEBO" "SyneTune" "DEHB"
#     do
#         echo "bash scripts/build_env.sh $benchmark $optimizer 3.11 \$ENV_LOCATION"
#     done
# done
