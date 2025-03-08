"""Random Search Optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from carps.optimizers.optimizer import Optimizer
from carps.utils.pareto_front import pareto
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent, SearchSpace


class RandomSearchOptimizer(Optimizer):
    """Random Search Optimizer."""

    def __init__(
        self,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize Random Search Optimizer.

        Parameters
        ----------
        task : Task
            The task (objective function with specific input and output space and optimization resources) to optimize.
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None.
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

        self.configspace: ConfigurationSpace = self.task.objective_function.configspace
        self.history: list[tuple[TrialInfo, TrialValue]] = []
        self.is_multifidelity = task.input_space.fidelity_space.is_multifidelity

        if hasattr(task, "n_objectives") and task.output_space.n_objectives is not None:
            self.is_multiobjective = task.output_space.n_objectives > 1

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        """Convert ConfigSpace's configuration space to the configuration space of the task.

        Returns:
        -------
        SearchSpace
            Search space of the task.
        """
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        """Convert a configuration to a trial info.

        Parameters
        ----------
        config : Configuration
            Configuration to convert.

        Returns:
        -------
        TrialInfo
            Trial info (config, seed, instance, budget).
        """
        budget = None
        if self.is_multifidelity:
            budget = self.task.input_space.fidelity_space.max_fidelity
            # budget = np.random.choice(np.linspace(self.task.input_space.fidelity_space.min_fidelity, self.task.input_space.fidelity_space.max_fidelity, 5))  # noqa: E501
        return TrialInfo(config=config, budget=budget)

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        Here, just sample a random configuration from the configuration space.

        Returns:
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        config = self.task.objective_function.configspace.sample_configuration()
        return self.convert_to_trial(config=config)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        In this case, just add the trial info and trial value to the history.

        Parameters
        ----------
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        self.history.append((trial_info, trial_value))

    def _setup_optimizer(self) -> None:
        return None

    def get_pareto_front(self) -> list[tuple[TrialInfo, TrialValue]]:
        """Return the pareto front for multi-objective optimization.

        Returns:
        -------
        list[tuple[TrialInfo, TrialValue]]
            Pareto front containing trial info and trial value.
        """
        if self.task.input_space.fidelity_space.is_multifidelity:
            max_fidelity = np.max([v[0].budget for v in self.history])
            results_on_highest_fidelity = np.array([v for v in self.history if v[0].budget == max_fidelity])
            costs = np.array([v[1].cost for v in results_on_highest_fidelity])
            # Determine pareto front of the trials run on max budget
            front = results_on_highest_fidelity[pareto(costs)]
        else:
            costs = np.array([v[1].cost for v in self.history])
            front = np.array(self.history)[pareto(costs)]
        return front.tolist()

    def get_current_incumbent(self) -> Incumbent:
        """Return the current incumbent.

        The incumbent is the current best configuration.
        In the case of multi-objective, there are multiple best configurations, mostly
        the Pareto front.

        Returns:
        -------
        Incumbent
            Incumbent tuple(s) containing trial info and trial value.
        """
        if self.task.output_space.n_objectives == 1:
            incumbent_tuple = min(self.history, key=lambda x: x[1].cost)
        else:
            incumbent_tuple = self.get_pareto_front()  # type: ignore[assignment]
        return incumbent_tuple
