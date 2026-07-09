"""Helper for papenbench configs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from carps.objective_functions.papenbench import get_benchmark_registry
from carps.utils.generate_tasks import (
    get_dict_input_space,
    get_dict_metadata,
    get_dict_opt_resources,
    get_dict_output_space,
)
from carps.utils.task import (
    FidelitySpace,
    InputSpace,
    OptimizationResources,
    OutputSpace,
    TaskMetadata,
    get_search_space_info,
)

os.environ["PATH"] = "/opt/software/pc2/EB-SW/software/Apptainer/1.3.5-GCCcore-13.3.0/bin:" + os.environ.get("PATH", "")


# Formula from YAHPO paper
def get_n_trials(dimension: int) -> int:
    """YAHPO n_trials formula."""
    return int(np.ceil(20 + 40 * np.sqrt(dimension)))


registry = get_benchmark_registry()

benchmark_id = "PapenBench"
objective_function_class = "carps.objective_functions.papenbench.PapenBenchObjectiveFunction"


def get_domain(benchmark_name: str) -> str:
    """Get domain by benchmark name."""
    return "TODO"
    if "lasso" in benchmark_name:
        return ...
    return ...


def get_obj_fun_approx(benchmark_name: str) -> str:
    """Get objective function approximation type by benchmark name."""
    return "TODO"
    if "lasso" in benchmark_name:
        ...
    return ...


for benchmark_name, benchmark_info in registry.items():
    print(benchmark_name, benchmark_info)
    task_id = f"{benchmark_id}/{benchmark_name}"

    target_path = Path("task")
    fn = target_path / (task_id + ".yaml")
    fn.parent.mkdir(exist_ok=True, parents=True)
    if fn.exists():
        print(f"-> Skipping {benchmark_name}: YAML configuration already exists at {fn}")
        continue

    objective_function_cfg = DictConfig(
        {
            "_target_": objective_function_class,
            "benchmark_name": benchmark_name,
        },
    )
    try:
        objective_function = instantiate(objective_function_cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Skipping {benchmark_name} due to error: {e}")
        continue

    search_space_kwargs = get_search_space_info(configspace=objective_function.configspace)

    n_trials = get_n_trials(len(objective_function.configspace))
    n_objectives = 1
    objectives = [f"objective_{i}" for i in range(n_objectives)]

    input_space = InputSpace(configuration_space=objective_function.configspace, fidelity_space=FidelitySpace())
    output_space = OutputSpace(
        n_objectives=n_objectives,
        objectives=objectives,  # type: ignore[arg-type]
    )
    optimization_resources = OptimizationResources(
        n_trials=n_trials,
        time_budget=None,
        n_workers=1,
    )

    task_metadata = TaskMetadata(
        has_constraints=False,
        domain=get_domain(benchmark_name),
        objective_function_approximation=get_obj_fun_approx(benchmark_name),
        has_virtual_time=False,
        deterministic=True,
        **search_space_kwargs,
    )

    cfg = DictConfig(
        {
            "benchmark_id": benchmark_id,
            "task_id": task_id,
            "task": {
                "_target_": "carps.utils.task.Task",
                "name": task_id,
                "seed": "${seed}",
                "objective_function": objective_function_cfg,
                "input_space": get_dict_input_space(input_space),
                "output_space": get_dict_output_space(output_space),
                "optimization_resources": get_dict_opt_resources(optimization_resources),
                "metadata": get_dict_metadata(task_metadata),
            },
        }
    )
    del objective_function

    yaml_str = OmegaConf.to_yaml(cfg=cfg)
    yaml_str = "# @package _global_\n" + yaml_str
    fn.write_text(yaml_str)
    print(cfg)
    print(fn)
