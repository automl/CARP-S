CONTAINER_FILE=$(python -c "from carps.objective_functions.papenbench import PAPENBENCH_CONTAINER_FILE; print(PAPENBENCH_CONTAINER_FILE)")
apptainer build  ${CONTAINER_FILE} ../container/recipes/benchmarks/PapenBench/papenbench.def
