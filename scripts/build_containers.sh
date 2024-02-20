# --------------------------------------------------------------------------------------------
# HANDLERS (mandatory)
# --------------------------------------------------------------------------------------------
./compile_noctua2.sh hydra_initializer.sif container_recipes/hydra_initializer.recipe

# --------------------------------------------------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------------------------------------------------
# Dummy Optimizer
./compile_noctua2.sh DUMMY_Optimizer.sif container_recipes/DUMMY_Optimizer/DUMMY_Optimizer.recipe

# Random Search
./compile_noctua2.sh RandomSearch.sif container_recipes/optimizer/randomsearch/randomsearch.recipe 

# SMAC3-2.0
./compile_noctua2.sh SMAC3-BlackBoxFacade.sif container_recipes/optimizer/smac20/smac20.recipe 

# SMAC3-1.4
./compile_noctua2.sh SMAC3-1.4-BlackBoxFacade.sif container_recipes/optimizer/smac14/smac14.recipe 

# --------------------------------------------------------------------------------------------
# PROBLEMS
# --------------------------------------------------------------------------------------------
# Dummy Problem
./compile_noctua2.sh DUMMY_Problem.sif container_recipes/DUMMY_Problem/DUMMY_Problem.recipe


