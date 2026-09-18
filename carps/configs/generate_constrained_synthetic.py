"""Helper for constrained synthetic task configs.

The benchmarks the constrained Bayesian optimization literature uses. Each config records the best
feasible objective value, computed from the problem definition rather than copied from the paper, so
that the number always matches the functions it is generated from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from carps.objective_functions.constrained_synthetic import PROBLEMS, ConstrainedSyntheticObjectiveFunction
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


# Formula from YAHPO paper
def get_n_trials(dimension: int) -> int:
    """YAHPO n_trials formula."""
    return int(np.ceil(20 + 40 * np.sqrt(dimension)))


benchmark_id = "ConstrainedSynthetic"
objective_function_class = "carps.objective_functions.constrained_synthetic.ConstrainedSyntheticObjectiveFunction"

target_path = Path("carps/configs/task/ConstrainedSynthetic")
target_path.mkdir(parents=True, exist_ok=True)

for problem_name, problem in PROBLEMS.items():
    objective_function = ConstrainedSyntheticObjectiveFunction(problem_name=problem_name, seed=0)
    search_space_kwargs = get_search_space_info(configspace=objective_function.configspace)

    task_id = f"ConstrainedSynthetic/{problem_name}"
    objective_function_cfg = {
        "_target_": objective_function_class,
        "problem_name": problem_name,
        "seed": "${seed}",
    }

    input_space = InputSpace(configuration_space=objective_function.configspace, fidelity_space=FidelitySpace())
    output_space = OutputSpace(n_objectives=1, objectives=["objective_0"])
    optimization_resources = OptimizationResources(n_trials=get_n_trials(problem.dimensions))
    task_metadata = TaskMetadata(
        has_constraints=True,
        domain="synthetic",
        objective_function_approximation="real",
        has_virtual_time=False,
        deterministic=True,
        **search_space_kwargs,
    )

    cfg = DictConfig(
        {
            "benchmark_id": benchmark_id,
            "task_id": task_id,
            "constraints": problem.constraint_expressions,
            # The best feasible objective value, for reporting a gap rather than a raw cost.
            "constrained_optimum": problem.constrained_optimum(),
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

    yaml_str = "# @package _global_\n" + OmegaConf.to_yaml(cfg=cfg)
    fn = target_path / f"{problem_name}.yaml"
    fn.write_text(yaml_str)
    print(f"{problem_name}: {problem.dimensions} vars, optimum {cfg.constrained_optimum:.6f} -> {fn}")
