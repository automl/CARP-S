from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from carps.optimizers.optimizer import Optimizer
from carps.utils.pareto_front import pareto
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent, SearchSpace


class RandomSearchOptimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)

        self.configspace: ConfigurationSpace = self.problem.configspace
        self.history: list[tuple[TrialInfo, TrialValue]] = []
        self.is_multifidelity = task.is_multifidelity

        if hasattr(task, "n_objectives"):
            self.is_multiobjective = task.n_objectives > 1

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        budget = None
        if self.is_multifidelity:
            budget = self.task.max_budget
            # budget = np.random.choice(np.linspace(self.task.min_budget, self.task.max_budget, 5))
        return TrialInfo(config=config, budget=budget)

    def ask(self) -> TrialInfo:
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config=config)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info, trial_value))

    def _setup_optimizer(self) -> None:
        return None

    def get_pareto_front(self) -> list[tuple[TrialInfo, TrialValue]]:
        """Return the pareto front for multi-objective optimization."""
        if self.task.is_multifidelity:
            max_budget = np.max([v[0].budget for v in self.history])
            results_on_highest_fidelity = np.array([v for v in self.history if v[0].budget == max_budget])
            costs = np.array([v[1].cost for v in results_on_highest_fidelity])
            # Determine pareto front of the trials run on max budget
            front = results_on_highest_fidelity[pareto(costs)]
        else:
            costs = np.array([v[1].cost for v in self.history])
            front = np.array(self.history)[pareto(costs)]
        return front.tolist()

    def get_current_incumbent(self) -> Incumbent:
        if self.task.n_objectives == 1:
            incumbent_tuple = min(self.history, key=lambda x: x[1].cost)
        else:
            incumbent_tuple = self.get_pareto_front()
        return incumbent_tuple
