export CONT_GENERAL_PATH=containers/general
export CONT_GENERAL_RECIPE_PATH=container_recipes/general
export CONT_BENCH_PATH=containers/benchmarks
export CONT_BENCH_RECIPE_PATH=container_recipes/benchmarks
export CONT_OPT_PATH=containers/optimizers
export CONT_OPT_RECIPE_PATH=container_recipes/optimizers

mkdir -p $CONT_GENERAL_PATH
mkdir -p $CONT_BENCH_PATH
mkdir -p $CONT_OPT_PATH

# --------------------------------------------------------------------------------------------
# HANDLERS (mandatory)
# --------------------------------------------------------------------------------------------
./scripts/compile_noctua2.sh ${CONT_GENERAL_PATH}/runner.sif ${CONT_GENERAL_RECIPE_PATH}/runner.recipe

# --------------------------------------------------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------------------------------------------------
# Dummy Optimizer
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/DUMMY_Optimizer.sif ${CONT_OPT_RECIPE_PATH}/DUMMY_Optimizer/DUMMY_Optimizer.recipe

# Random Search
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/RandomSearch.sif ${CONT_OPT_RECIPE_PATH}/RandomSearch/RandomSearch.recipe 

# SMAC3-2.0
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/SMAC3.sif ${CONT_OPT_RECIPE_PATH}/SMAC3/SMAC3.recipe 

# SMAC3-1.4
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/SMAC3-1.4.sif ${CONT_OPT_RECIPE_PATH}/SMAC3-1.4/SMAC3-1.4.recipe 

# HEBO
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/HEBO.sif ${CONT_OPT_RECIPE_PATH}/HEBO/HEBO.recipe 

# SyneTune
./scripts/compile_noctua2.sh ${CONT_OPT_PATH}/SyneTune.sif ${CONT_OPT_RECIPE_PATH}/SyneTune/SyneTune.recipe 

# --------------------------------------------------------------------------------------------
# PROBLEMS
# --------------------------------------------------------------------------------------------
# Dummy Problem
./scripts/compile_noctua2.sh ${CONT_BENCH_PATH}/DUMMY_Problem.sif ${CONT_BENCH_RECIPE_PATH}/DUMMY_Problem/DUMMY_Problem.recipe

# HPOB
./scripts/compile_noctua2.sh ${CONT_BENCH_PATH}/HPOB.sif ${CONT_BENCH_RECIPE_PATH}/HPOB/HPOB.recipe ${CONT_BENCH_RECIPE_PATH}/HPOB/download_data.sh

# YAHPO
./scripts/compile_noctua2.sh ${CONT_BENCH_PATH}/YAHPO.sif ${CONT_BENCH_RECIPE_PATH}/YAHPO/YAHPO.recipe


