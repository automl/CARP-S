from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json



@dataclass_json
@dataclass(frozen=True)
class Task():
    # General
    dimensions: int | None = None
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
    search_space_has_conditionals: bool | None = None
    search_space_has_categoricals: bool | None = None
    search_space_has_forbiddens: bool | None = None
    search_space_has_priors: bool | None = None

    def __post_init__(self):
        if self.is_multifidelity and self.max_budget is None:
            raise ValueError("Please specify max budget for multifidelity.")
        if self.n_trials is None and self.time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget`.")