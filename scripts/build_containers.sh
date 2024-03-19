export CONT_GENERAL_PATH=containers/general
export CONT_GENERAL_RECIPE_PATH=container_recipes/general
export CONT_BENCH_PATH=containers/benchmarks
export CONT_BENCH_RECIPE_PATH=container_recipes/benchmarks
export CONT_OPT_PATH=containers/optimizers
export CONT_OPT_RECIPE_PATH=container_recipes/optimizers

# --------------------------------------------------------------------------------------------
# HANDLERS (mandatory)
# --------------------------------------------------------------------------------------------
./compile_noctua2.sh ${CONT_GENERAL_PATH}/exp_config_generator.sif ${CONT_GENERAL_RECIPE_PATH}/exp_config_generator.recipe

# --------------------------------------------------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------------------------------------------------
# Dummy Optimizer
./compile_noctua2.sh ${CONT_OPT_PATH}/DUMMY_Optimizer.sif ${CONT_OPT_RECIPE_PATH}/DUMMY_Optimizer/DUMMY_Optimizer.recipe

# Random Search
./compile_noctua2.sh ${CONT_OPT_PATH}/RandomSearch.sif ${CONT_OPT_RECIPE_PATH}/randomsearch/randomsearch.recipe 

# SMAC3-2.0
./compile_noctua2.sh ${CONT_OPT_PATH}/SMAC3.sif ${CONT_OPT_RECIPE_PATH}/smac20/smac20.recipe 

# SMAC3-1.4
./compile_noctua2.sh ${CONT_OPT_PATH}/SMAC3-1.4.sif ${CONT_OPT_RECIPE_PATH}/smac14/smac14.recipe 

# HEBO
./compile_noctua2.sh ${CONT_OPT_PATH}/HEBO.sif ${CONT_OPT_RECIPE_PATH}/HEBO/HEBO.recipe 

# --------------------------------------------------------------------------------------------
# PROBLEMS
# --------------------------------------------------------------------------------------------
# Dummy Problem
./compile_noctua2.sh ${CONT_BENCH_PATH}/DUMMY_Problem.sif ${CONT_BENCH_RECIPE_PATH}/DUMMY_Problem/DUMMY_Problem.recipe

# HPOB
./compile_noctua2.sh ${CONT_BENCH_PATH}/HPOB.sif ${CONT_BENCH_RECIPE_PATH}/hpob/hpob_container.recipe ${CONT_BENCH_RECIPE_PATH}/hpob/download_data.sh


