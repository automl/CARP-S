from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, Constant, OrdinalHyperparameter
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dataclasses_json import dataclass_json


def get_search_space_info(configspace: ConfigurationSpace) -> dict[str, Any]:
    hps = dict(configspace)
    dimension = len(hps)
    search_space_has_priors = False
    n_categoricals = 0
    n_integers = 0
    n_ordinals = 0
    n_floats = 0
    for hp in hps.values():
        if isinstance(hp, CategoricalHyperparameter):
            n_categoricals += 1
        elif isinstance(hp, BetaIntegerHyperparameter | NormalIntegerHyperparameter | UniformIntegerHyperparameter):
            n_integers += 1
        elif isinstance(hp, BetaFloatHyperparameter | NormalFloatHyperparameter | UniformFloatHyperparameter):
            n_floats += 1
        elif isinstance(hp, OrdinalHyperparameter):
            n_ordinals += 1
        elif isinstance(hp, Constant):
            dimension -= 1

        if isinstance(
            hp,
            BetaFloatHyperparameter
            | BetaIntegerHyperparameter
            | NormalFloatHyperparameter
            | NormalIntegerHyperparameter,
        ):
            search_space_has_priors = True

    assert n_categoricals + n_floats + n_integers + n_ordinals == dimension

    search_space_has_conditionals = len(configspace.get_conditions()) > 0
    search_space_has_forbiddens = len(configspace.get_forbiddens()) > 0
    return {
        "dimensions": dimension,
        "search_space_n_categoricals": n_categoricals,
        "search_space_n_ordinals": n_ordinals,
        "search_space_n_integers": n_integers,
        "search_space_n_floats": n_floats,
        "search_space_has_conditionals": search_space_has_conditionals,
        "search_space_has_forbiddens": search_space_has_forbiddens,
        "search_space_has_priors": search_space_has_priors,
    }


@dataclass_json
@dataclass(frozen=True)
class Task:
    """Task information.

    For general optimization, only `n_trials` or `time_budget` needs
    to be defined. The optimizers receive the search space.
    For multi-fidelity, at least `is_multifidelity` and `max_budget´
    need to be specified.
    For multi-objecitve, at least `n_objectives` needs to be specified.
    The remaining parameters are meta-data and not necessarily needed
    by the optimizer but useful to order tasks.

    Parameters
    ----------
    # General
    n_trials : int
        The number of trials aka calls to the objective function.
        Specify this for classic blackbox problems.
        Either `n_trials´ or `time_budget` needs to be specified.
    time_budget : float
        The time budget in minutes for optimization.
        Specify this for multi-fidelity problems.
        Either `n_trials´ or `time_budget` needs to be specified.

    # Parallelism
    n_workers : int = 1
        The number of workers allowed for this task. Not every optimizer
        allows parallelism.

    # Multi-objective
    n_objectives : int
        The number of optimization objectives.
    objectives : list[str]
        Optional names of objectives.

    # Multi-fidelity
    is_multifidelity : bool
        Whether the task is a multi-fidelity problem.
    fidelity_type : str
        The kind of fidelity used.
    min_budget : float
        Minimum fidelity. Not used by every optimizer.
    max_budget : float
        Maximum fidelity. Required for multi-fidelity.

    has_constraints : bool
        Whether the task has any constraints.

    # Objective Function Characteristics
    domain : str
        The task's domain, e.g. synethetic, ML, NAS.
    objective_function_approximation : str
        How the objective function is approximated / represented, e.g.
        real, surrogate or tabular.
    has_virtual_time : bool
        Whether the task tracked evaluation time in the case of surrogate
        or tabular objective functions.
    deterministic : bool
        Whether the objective function is deterministic.

    # Search Space Information
    dimensions: int
        The dimensionality of the task.
    search_space_n_categoricals: int
        The number of categorical hyperparameters (HPs).
    search_space_n_ordinals: int
        The number of ordinal HPs.
    search_space_n_integers: int
        The number of integer HPs.
    search_space_n_floats: int
        The number of float HPs.
    search_space_has_conditionals: bool
        Whether the search space has conditions. Not every optimizer
        supports conditional search spaces.
    search_space_has_forbiddens: bool
        Whether the search space has forbiddens/constraints.
        Not every optimizer supports forbiddens.
    search_space_has_priors: bool
        Whether there are any priors on HPs, e.g. beta or normal.

    Raises:
    ------
    ValueError
        When `is_multifidelity` is set and `max_budget` not specified.
        In order to use multi-fidelity, both need to be specified.
    ValueError
        When neither `n_trials` nor `time_budget` are specified.
    """

    # General (REQUIRED)
    n_trials: int | None = None
    time_budget: float | None = None  # 1 cpu, walltime budget in minutes

    # Parallelism
    n_workers: int = 1

    # Multi-Objective
    n_objectives: int | None = None
    objectives: list[str] | None = None

    # Multi-Fidelity
    is_multifidelity: bool | None = None
    fidelity_type: str | None = None
    min_budget: float | None = None
    max_budget: float | None = None

    # Constraint BO
    has_constraints: bool | None = None

    # Objective Function Characteristics
    domain: str | None = None  # e.g. synthetic, ML, NAS, x
    objective_function_approximation: str | None = None  # real, surrogate, tabular
    has_virtual_time: bool | None = None
    deterministic: bool | None = None

    # Search Space
    dimensions: int | None = None
    search_space_n_categoricals: int | None = None
    search_space_n_ordinals: int | None = None
    search_space_n_integers: int | None = None
    search_space_n_floats: int | None = None
    search_space_has_conditionals: bool | None = None
    search_space_has_forbiddens: bool | None = None
    search_space_has_priors: bool | None = None

    def __post_init__(self):
        if self.is_multifidelity and self.max_budget is None:
            raise ValueError("Please specify max budget for multifidelity.")
        if self.n_trials is None and self.time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget`.")
