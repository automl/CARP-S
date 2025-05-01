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
# bash scripts/install_yahpo.sh

bash scripts/build_env.sh SMAC3 HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3 YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3 HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh SMAC3 BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3 MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3-1.4 HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3-1.4 YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3-1.4 HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh SMAC3-1.4 BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SMAC3-1.4 MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh RandomSearch HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh RandomSearch YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh RandomSearch HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh RandomSearch BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh RandomSearch MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh HEBO HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh HEBO YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh HEBO HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh HEBO BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh HEBO MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh SyneTune HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SyneTune YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh SyneTune HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh SyneTune BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh SyneTune MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh DEHB HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh DEHB YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh DEHB HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh DEHB BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh DEHB MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh Optuna HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh Optuna YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh Optuna HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh Optuna BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh Optuna MFPBench 3.11 $ENV_LOCATION
bash scripts/build_env.sh Nevergrad HPOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh Nevergrad YAHPO 3.11 $ENV_LOCATION
bash scripts/build_env.sh Nevergrad HPOBench 3.9 $ENV_LOCATION
bash scripts/build_env.sh Nevergrad BBOB 3.11 $ENV_LOCATION
bash scripts/build_env.sh Nevergrad MFPBench 3.11 $ENV_LOCATION


# for optimizer in "SMAC3" "SMAC3-1.4" "RandomSearch" "HEBO" "SyneTune" "DEHB" "Optuna" "Nevergrad"
# do
#     for benchmark in "HPOB 3.11" "YAHPO 3.11" "HPOBench 3.9" "BBOB 3.11" "MFPBench 3.11"
#     do
#         echo "bash scripts/build_env.sh $optimizer $benchmark \$ENV_LOCATION"
#     done
# done
