"""Optuna optimizer."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import optuna
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from optuna.distributions import (  # type: ignore
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState as OptunaTrialState  # type: ignore
from rich import print as printr

from carps.optimizers.optimizer import Optimizer
from carps.utils.pareto_front import pareto
from carps.utils.trials import StatusType, TrialInfo, TrialValue

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from optuna.study import Study  # type: ignore

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent

# NOTE: Optuna has an extra OptunaTrialState.PRUNED, which indicates something
# was halted during it's run and not progressed to the next budget. They fundamentally
# do multi-fidelity differently in that they stop currently running trials and not restart
# a trial to run until a higher budget.
optuna_trial_states: dict[StatusType, OptunaTrialState] = {
    StatusType.RUNNING: OptunaTrialState.RUNNING,
    StatusType.SUCCESS: OptunaTrialState.COMPLETE,
    StatusType.CRASHED: OptunaTrialState.FAIL,
    StatusType.TIMEOUT: OptunaTrialState.FAIL,
    StatusType.MEMORYOUT: OptunaTrialState.FAIL,
}


def hp_to_optuna_distribution(hp: Hyperparameter) -> BaseDistribution:
    """Parse a ConfigSpace hyperparameter to an Optuna distribution.

    Parameters
    ----------
    hp : Hyperparameter
        The ConfigSpace hyperparameter to parse.

    Returns:
    -------
    BaseDistribution
        The Optuna distribution.
    """
    if isinstance(hp, UniformIntegerHyperparameter):
        return IntDistribution(hp.lower, hp.upper, log=hp.log)
    if isinstance(hp, UniformFloatHyperparameter):
        return FloatDistribution(hp.lower, hp.upper, log=hp.log)
    if isinstance(hp, CategoricalHyperparameter):
        if hp.weights is not None:
            raise NotImplementedError(f"Weights are not supported in Optuna ({hp})")

        return CategoricalDistribution(hp.choices)
    if isinstance(hp, OrdinalHyperparameter):
        warnings.warn(
            f"Ordinal hyperparameters are not supported in Optuna, use Categorical instead for {hp}.", stacklevel=2
        )
        return CategoricalDistribution(hp.sequence)
    if isinstance(hp, Constant):
        return CategoricalDistribution([hp.value])

    raise NotImplementedError(f"Can't handle hyperparameter {hp}")


class OptunaOptimizer(Optimizer):
    """An optimizer that uses Optuna to optimize a search space."""

    def __init__(
        self,
        task: Task,
        optuna_cfg: DictConfig,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the optimizer.

        Parameters
        ----------
        task : Task
            The task (objective function with specific input and output space and optimization resources) to optimize.
        optuna_cfg : DictConfig
            The configuration for the Optuna optimizer.
        loggers : list[AbstractLogger] | None
            The loggers to use during optimization.
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
        self._solver: Study | None = None
        self.optuna_cfg = optuna_cfg

        configspace = self.task.objective_function.configspace
        if any(configspace.forbidden_clauses):
            raise NotImplementedError("Forbidden clauses are not yet supported in Optuna")

        self.optuna_space = {hp.name: hp_to_optuna_distribution(hp) for hp in list(configspace.values())}
        self.configspace = configspace
        self.history: dict[str, tuple[optuna.Trial, Configuration, None | float | list[float]]] = {}

    def _setup_optimizer(self) -> optuna.study.Study:
        # TODO: minimize always?
        # TODO How do I know if it's multi-objective?

        # if multiple_metrics:
        #     sampler = NSGAIISampler(seed=self.optuna_cfg.sampler.seed)
        # else
        #     sampler = TPESampler(seed=self.optuna_cfg.sampler.seed)
        """(function) def create_study(
            *,
            storage: str | BaseStorage | None = None,
            sampler: BaseSampler | None = None,
            pruner: BasePruner | None = None,
            study_name: str | None = None,
            direction: str | StudyDirection | None = None,
            load_if_exists: bool = False,
            directions: Sequence[str | StudyDirection] | None = None
        ) -> Study.
        """
        study = optuna.create_study(
            **self.optuna_cfg.study, directions=["minimize"] * self.task.output_space.n_objectives
        )
        printr(study)

        return study

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Returns:
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        assert self._solver is not None

        optuna_trial: optuna.Trial = self._solver.ask(self.optuna_space)
        config = optuna_trial.params
        trial_number = optuna_trial.number
        unique_name = f"{trial_number=}"
        configspace_config = Configuration(
            configuration_space=self.configspace, values=config, allow_inactive_with_values=True
        )
        self.history[unique_name] = (optuna_trial, configspace_config, None)
        return TrialInfo(
            config=configspace_config,
            name=unique_name,
            instance=None,
            budget=None,
            seed=None,
        )

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        unique_name = trial_info.name
        assert unique_name is not None
        assert unique_name in self.history
        assert self._solver is not None

        optuna_status = optuna_trial_states[trial_value.status]

        # NOTE: Is there any case in which StatusType.RUNNING is returned, giving
        # us this OptunaTrialState.RUNNING?
        assert optuna_status is not OptunaTrialState.RUNNING

        optuna_trial, configspace_config, prev_cost = self.history[unique_name]
        assert prev_cost is None

        # Need to update the history with the cost such that we can access the cost
        # for `get_current_incumbent()`
        cost = trial_value.cost if optuna_status is OptunaTrialState.COMPLETE else None
        self.history[unique_name] = (optuna_trial, configspace_config, cost)

        self._solver.tell(trial=optuna_trial, values=cost, state=optuna_status)

    def get_pareto_front(self) -> list[tuple[Configuration, None | float | list[float]]]:
        """Return the pareto front for multi-objective optimization.

        Returns:
        -------
        list[tuple[Configuration, None | float | list[float]]]
            The pareto front as a list of tuples of configuration and cost.
        """
        non_none_entries = [[config, cost] for optuna_trial, config, cost in self.history.values() if cost is not None]
        costs = np.array([v[1] for v in non_none_entries])
        ids_bool = pareto(costs)
        ids = np.where(ids_bool)[0]
        pareto_front: list[tuple[Configuration, None | float | list[float]]] = [non_none_entries[i] for i in ids]  # type: ignore[misc]
        return pareto_front

    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        if self.task.output_space.n_objectives == 1:
            non_none_entries = [(config, cost) for _, config, cost in self.history.values() if cost is not None]
            if len(non_none_entries) == 0:
                return None
            best_config, best_cost = min(non_none_entries, key=lambda x: x[1])
            assert not isinstance(best_cost, Iterable)
            trial_info = TrialInfo(config=best_config)
            trial_value = TrialValue(cost=best_cost)
            incumbent_tuple = (trial_info, trial_value)
        else:
            front = self.get_pareto_front()
            incumbent_tuple = [  # type: ignore[assignment]
                (TrialInfo(config=config), TrialValue(cost=cost)) for config, cost in front if cost is not None
            ]
        return incumbent_tuple

    # NOT really needed
    def convert_configspace(self, configspace: ConfigurationSpace) -> dict[str, BaseDistribution]:
        """Convert ConfigSpace configuration space to search space from optimizer.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from ObjectiveFunction.

        Returns:
        -------
        SearchSpace
            Optimizer's search space.
        """
        raise NotImplementedError

    # NOT really needed
    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the objective function can be evaluated with a unified API.

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        raise NotImplementedError
