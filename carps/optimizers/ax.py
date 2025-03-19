"""Ax Optimizer.

* Source: https://github.com/facebook/Ax

* Paper:
E. Bakshy, L. Dworkin, B. Karrer, K. Kashin, B. Letham, A. Murthy, S. Singh.
AE: A domain-agnostic platform for adaptive experimentation. Workshop on Systems
for ML and Open Source Software, NeurIPS 2018.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.instantiation import InstantiationBase
from ax.utils.common.random import set_rng_seed
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ax.core.search_space import SearchSpace
    from ax.core.types import (
        TParameterization,
        TParamValue,
    )
    from ConfigSpace import ConfigurationSpace
    from omegaconf import DictConfig

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


def configspace2ax(name: str, parameter: Hyperparameter) -> dict[str, TParamValue | Sequence[TParamValue]]:
    """Converts a ConfigSpace hyperparameter into a suitable Ax representation.

    Parameters
    ----------
    name : str
        Hyperparameter name
    parameter : Hyperparameter
        Hyperparameter to convert

    Returns:
    -------
    dict[str, TParamValue | Sequence[TParamValue]]
        Hyperparameter representation suitable for Ax's search space
    """
    res: dict[str, TParamValue | Sequence[TParamValue]] = {}

    res["name"] = name
    res["log_scale"] = parameter.log if hasattr(parameter, "log") else False

    if isinstance(parameter, IntegerHyperparameter):
        res["value_type"] = "int"
        res["type"] = "range"
        res["bounds"] = [parameter.lower, parameter.upper]

    elif isinstance(parameter, FloatHyperparameter):
        res["value_type"] = "float"
        res["type"] = "range"
        res["bounds"] = [parameter.lower, parameter.upper]

    elif isinstance(parameter, CategoricalHyperparameter):
        res["type"] = "choice"
        res["values"] = list(parameter.choices)

    elif isinstance(parameter, Constant):
        res["type"] = "fixed"
        res["value"] = parameter.value

    elif isinstance(parameter, OrdinalHyperparameter):
        res["is_ordered"] = True
        res["type"] = "choice"
        res["values"] = list(parameter.sequence)

    else:
        raise NotImplementedError("Invalid hyperparameter found during instantiation of Ax search space")

    return res


class AxOptimizer(Optimizer):
    """Ax optimizer. Supports single- and multi-objective tasks.

    Random Seed
    -------
    Note that this setting only affects the Sobol quasi-random generator
    and BoTorch-powered Bayesian optimization models. For the latter models,
    setting random seed to the same number for two optimizations will make
    the generated trials similar, but not exactly the same, and over time
    the trials will diverge more.
    """

    def __init__(
        self,
        task: Task,
        ax_cfg: DictConfig,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the AxOptimizer.

        Parameters
        ----------
        task : Task
            The task (objective function with specific input and output space and optimization resources) to optimize.
        ax_cfg : DictConfig
            (Hydra) Configuration for Ax.
        loggers : list[AbstractLogger] | None, optional
            A list of loggers for tracking. Defaults to None.
        expects_multiple_objectives : bool, optional
            Metadata. Whether the optimizer expects multiple objectives, by default False.
        expects_fidelities : bool, optional
            Metadata. Whether the optimizer expects fidelities for multi-fidelity, by default False.
        """
        super().__init__(
            task,
            loggers,
            expects_fidelities=expects_fidelities,
            expects_multiple_objectives=expects_multiple_objectives,
        )

        self._parameters: list[dict[str, TParamValue | Sequence[TParamValue]]] = []
        self._parameter_constraints: list[str] = []

        self.task = task
        self.ax_configspace = self.convert_configspace(self.task.objective_function.configspace)
        self.ax_cfg = ax_cfg
        self._solver: AxClient | None = None
        self.history: dict[str, dict[str, Any]] = {}

    def _setup_optimizer(self) -> Any:
        ax_client = AxClient(random_seed=self.ax_cfg.scenario.seed)

        ax_client.create_experiment(
            parameters=self._parameters,
            parameter_constraints=self._parameter_constraints,
            objectives={
                objective: ObjectiveProperties(minimize=True) for objective in self.ax_cfg.scenario.objectives
            },  # Note: Always minimization objective
        )

        return ax_client

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        """Converts a ConfigurationSpace into Ax's SearchSpace representation.

        Parameters
        ----------
        configspace : ConfigurationSpace
            ConfigurationSpace to convert

        Returns:
        -------
        SearchSpace
            SearchSpace representaton of the input configspace
        """
        for name, parameter in configspace.items():
            self._parameters.append(configspace2ax(name, parameter))

        return InstantiationBase.make_search_space(
            parameters=self._parameters, parameter_constraints=self._parameter_constraints
        )

    def convert_to_trial(  # type: ignore[override]
        self, trial: TParameterization, trial_index: int
    ) -> TrialInfo:
        """Converts Ax's TParameterization to CARPS TrialInfo.

        Parameters
        ----------
        trial: TParameterization
            Ax trial to convert
        trial_index: int
            Trial index

        Returns:
        -------
        TrialInfo
            TrialInfo representation of the input trial
        """
        # Allow inactivate parameter values for optimizers that cannot handle conditions
        # In that case they will propose a value for each HP, whether it is active or not.
        config = Configuration(self.task.objective_function.configspace, values=trial, allow_inactive_with_values=True)

        return TrialInfo(config=config, seed=self.ax_cfg.scenario.seed, name=str(trial_index))

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        Returns:
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        set_rng_seed(self.ax_cfg.scenario.seed)  # Note: This does not guarantee deterministic behavior
        parameterization, trial_index = self.solver.get_next_trial()

        return self.convert_to_trial(parameterization, trial_index)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        if len(self.ax_cfg.scenario.objectives) == 1:
            raw_data = {objective: trial_value.cost for objective in self.ax_cfg.scenario.objectives}
        else:
            assert isinstance(trial_value.cost, list)
            raw_data = {objective: trial_value.cost[i] for i, objective in enumerate(self.ax_cfg.scenario.objectives)}

        trial_index = int(trial_info.name) if trial_info.name is not None else -1

        self.solver.complete_trial(trial_index=trial_index, raw_data=raw_data)

    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        if len(self.ax_cfg.scenario.objectives) == 1:
            index, best_parameters, values = self.solver.get_best_trial()

            cost = values[0]
        else:
            res = self.solver.get_pareto_optimal_parameters()

            index = next(iter(res.keys()))
            best_parameters = res[index][0]
            values = res[index][1][1]

            cost = {k: values[k][k] for k in values}

        inc_config = self.convert_to_trial(best_parameters, index)

        inc_value = TrialValue(cost=cost)
        return (inc_config, inc_value)
