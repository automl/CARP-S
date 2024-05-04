from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter, Constant
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    NormalIntegerHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
)
from typing import Any


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
        elif isinstance(hp, (BetaIntegerHyperparameter, NormalIntegerHyperparameter, UniformIntegerHyperparameter)):
            n_integers += 1
        elif isinstance(hp, (BetaFloatHyperparameter, NormalFloatHyperparameter, UniformFloatHyperparameter)):
            n_floats += 1
        elif isinstance(hp, OrdinalHyperparameter):
            n_ordinals += 1
        elif isinstance(hp, Constant):
            dimension -= 1

        if isinstance(hp, (BetaFloatHyperparameter, BetaIntegerHyperparameter, NormalFloatHyperparameter, NormalIntegerHyperparameter)):
            search_space_has_priors = True

    assert n_categoricals + n_floats + n_integers + n_ordinals == dimension
    
    search_space_has_conditionals = len(configspace.get_conditions()) > 0
    search_space_has_forbiddens = len(configspace.get_forbiddens()) > 0
    search_space_info = {
        "dimensions": dimension,
        "search_space_n_categoricals": n_categoricals,
        "search_space_n_ordinals": n_ordinals,
        "search_space_n_integers": n_integers,
        "search_space_n_floats": n_floats,
        "search_space_has_conditionals": search_space_has_conditionals,
        "search_space_has_forbiddens": search_space_has_forbiddens,
        "search_space_has_priors": search_space_has_priors,
    }
    return search_space_info



@dataclass_json
@dataclass(frozen=True)
class Task():
    # General
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