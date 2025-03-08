"""Task Definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from omegaconf import ListConfig

from carps.utils.loggingutils import get_logger

if TYPE_CHECKING:
    from carps.objective_functions.objective_function import ObjectiveFunction

input_space_logger = get_logger("InputSpace")
output_space_logger = get_logger("OutputSpace")


def get_search_space_info(configspace: ConfigurationSpace) -> dict[str, Any]:
    """Get info about the search space.

    Number of dimensions, which hyperparameters and whether it has conditions or forbiddens.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space.

    Returns.
    -------
    dict[str, Any]
        Info dict with keys
        dimensions, search_space_n_categoricals, search_space_n_ordinals, search_space_n_integers,
        search_space_n_floats, search_space_has_conditionals,
        search_space_has_forbiddens, search_space_has_priors
    """
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
class TaskMetadata:
    """Task metadata.

    Parameters
    ----------
    # Constraint BO
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

    """

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


@dataclass_json
@dataclass(frozen=True)
class FidelitySpace:
    """Fidelity Space.

    Determines if and how multi-fidelity optimization should be performed.

    Parameters
    ----------
    is_multifidelity : bool
        Whether the task is a multi-fidelity optimization task.
    fidelity_type : str | None
        The type of fidelity, e.g. time, memory, etc.
    min_fidelity : int | float | None
        The minimum budget.
    max_fidelity : int | float | None
        The maximum budget.
    """

    is_multifidelity: bool = False
    fidelity_type: str | None = None
    min_fidelity: int | float | None = None
    max_fidelity: int | float | None = None

    def __post_init__(self) -> None:
        if self.is_multifidelity:
            assert self.fidelity_type is not None
            assert self.min_fidelity is not None
            assert self.max_fidelity is not None


@dataclass_json
@dataclass(frozen=True)
class InputSpace:
    """Input Space.

    This is the input to the objective function.
    In general, only the configuration space is subject to optimization.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
        The configuration space.
    fidelity_space : FidelitySpace | None
        The fidelity space. Determines if and how multi-fidelity optimization should be performed.
    instance_space : Any | None
        The instance space. This is relevant for algorithm configuration.
    """

    configuration_space: ConfigurationSpace
    fidelity_space: FidelitySpace = field(default_factory=FidelitySpace)
    instance_space: Any | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.configuration_space, ConfigurationSpace)


@dataclass_json
@dataclass(frozen=True)
class OptimizationResources:
    """Optimization Resources.

    Parameters
    ----------
    n_trials : int | None
        The number of trials (objective function evaluations) (full trials == at the highest fidelity in the case of
        multi-fidelity).
    time_budget : float | None
        The time budget in minutes.
    n_workers : int
        The number of workers.
    """

    n_trials: int | None = None
    time_budget: float | None = None  # 1 cpu, walltime budget in minutes

    # Parallelism
    n_workers: int = 1

    def __post_init__(self) -> None:
        if self.n_trials is None and self.time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget`.")


@dataclass_json
@dataclass(frozen=True)
class OutputSpace:
    """Output Space.

    This is the output of the objective function.

    Parameters
    ----------
    n_objectives : int
        The number of objectives.
    objectives : tuple[str]
        The names of the objectives.
    """

    # objective_space: ConfigurationSpace | None = None

    n_objectives: int = 1
    objectives: tuple[str] = ("quality",)

    # def __post_init__(self) -> None:
    #     # Set the number of objectives and their names from the objective space
    #     if self.objective_space is not None:
    #         object.__setattr__(self, "n_objectives", len(list(self.objective_space.values())))
    #         object.__setattr__(self, "objectives", tuple(self.objective_space.keys()))

    #     # If no objective space is specified, use a default unbounded space
    #     else:
    #         default_space = ConfigurationSpace()
    #         default_space.add(
    #             UniformFloatHyperparameter(
    #                 name="quality",
    #                 lower=VERY_SMALL_NUMBER,
    #                 upper=VERY_LARGE_NUMBER,
    #                 log=False,
    #             )
    #         )
    #         object.__setattr__(self, "objective_space", default_space)

    #         output_space_logger.info("No objective space specified. Using default unbounded space: "\
    #                                  f"{self.objective_space} ({self.n_objectives}, {self.objectives}).")

    def __post__init__(self) -> None:
        assert self.n_objectives == len(self.objectives)
        if isinstance(self.objectives, ListConfig):
            object.__setattr__(self, "objectives", tuple(self.objectives))


@dataclass_json
@dataclass(frozen=True)
class Task:
    """Task.

    The optimization task with optimization resources.

    Parameters
    ----------
    name : str
        The name of the task.
    objective_function : ObjectiveFunction
        The objective function.
    input_space : InputSpace
        The input space.
    output_space : OutputSpace
        The output space.
    optimization_resources : OptimizationResources
        The optimization resources.
    metadata : TaskMetadata
        The task metadata containing extra information like hyperparameter types.
    seed : int | None
        The seed.
    """

    name: str
    objective_function: ObjectiveFunction
    input_space: InputSpace
    output_space: OutputSpace
    optimization_resources: OptimizationResources
    metadata: TaskMetadata
    seed: int | None = None
