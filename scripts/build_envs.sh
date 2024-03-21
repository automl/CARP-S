# bash scripts/build_env.sh OPTIMIZER_CONTAINER_ID BENCHMARK_ID PYTHON_VERSION 
bash scripts/build_env.sh DUMMY_Optimizer DUMMY_Problem 3.11
bash scripts/build_env.sh DUMMY_Optimizer HPOB 3.11

bash scripts/build_env.sh RandomSearch DUMMY_Problem 3.11
bash scripts/build_env.sh RandomSearch HPOB 3.11

bash scripts/build_env.sh SMAC3 DUMMY_Problem 3.11
bash scripts/build_env.sh SMAC3 HPOB 3.11

bash scripts/build_env.sh SMAC3-1.4 DUMMY_Problem 3.11
bash scripts/build_env.sh SMAC3-1.4 HPOB 3.11
