"""Helper for constrained pymoo task configs.

pymoo ships a large family of constrained problems which were unreachable while
`PymooObjectiveFunction` could not read their constraint values. This writes a task config for every
single-objective one, alongside the existing unconstrained configs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymoo.problems
from omegaconf import DictConfig, OmegaConf

from carps.objective_functions.pymoo import PymooObjectiveFunction
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


benchmark_id = "Pymoo"
objective_function_class = "carps.objective_functions.pymoo.PymooObjectiveFunction"

# The single-objective constrained problems of the G-suite, plus the constrained engineering problems.
PROBLEM_NAMES = [f"g{i}" for i in range(1, 25)] + ["welded_beam", "pressure_vessel"]

target_path = Path("carps/configs/task/Pymoo/SO/constraint")
target_path.mkdir(parents=True, exist_ok=True)

for problem_name in PROBLEM_NAMES:
    try:
        problem = pymoo.problems.get_problem(problem_name)
    except Exception as e:  # noqa: BLE001
        print(f"Skipping {problem_name}: {e}")
        continue

    if problem.n_obj != 1:
        print(f"Skipping {problem_name}: {problem.n_obj} objectives, not single-objective.")
        continue

    if problem.n_ieq_constr == 0 and problem.n_eq_constr == 0:
        print(f"Skipping {problem_name}: no constraints.")
        continue

    if problem.n_eq_constr > 0:
        # An equality constraint cannot be written as a bound, and a method that models the probability of
        # satisfying a bound cannot represent one either. Emitting the inequality constraints alone would
        # describe a different, easier problem than the one the configuration names.
        print(f"Skipping {problem_name}: {problem.n_eq_constr} equality constraints cannot be written as bounds.")
        continue

    objective_function = PymooObjectiveFunction(problem_name=problem_name, seed=0)
    search_space_kwargs = get_search_space_info(configspace=objective_function.configspace)

    task_id = f"Pymoo/SO/constraint/{problem_name}"
    objective_function_cfg = {
        "_target_": objective_function_class,
        "problem_name": problem_name,
        "problem_kwargs": {},
        "seed": "${seed}",
    }

    input_space = InputSpace(configuration_space=objective_function.configspace, fidelity_space=FidelitySpace())
    output_space = OutputSpace(n_objectives=1, objectives=["objective_0"])
    optimization_resources = OptimizationResources(n_trials=get_n_trials(len(objective_function.configspace)))
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
            # Inequality constraints are satisfied at or below zero, equality constraints at zero. The names
            # match what PymooObjectiveFunction reports alongside the cost.
            "constraints": [f"g{i} <= 0.0" for i in range(problem.n_ieq_constr)],
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
    print(f"{problem_name}: {problem.n_var} vars, {problem.n_ieq_constr} ineq, {problem.n_eq_constr} eq -> {fn}")
