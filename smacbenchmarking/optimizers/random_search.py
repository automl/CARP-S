from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class RandomSearchOptimizer(Optimizer):
    def __init__(
            self,
            problem: Problem,
            n_trials: int | None,
            time_budget: float | None,
            n_workers: int = 1,
            loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, n_trials, time_budget, n_workers, loggers)

        self.configspace: ConfigurationSpace = self.problem.configspace
        self.n_trials: int = n_trials
        self.history: list[tuple[Configuration,float]] = []

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        return TrialInfo(config=config)
    
    def ask(self) -> TrialInfo:
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config=config)
    
    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info.config, trial_value.cost))

    def _setup_optimizer(self) -> None:
        return None
    
    def get_current_incumbent(self) -> tuple[Configuration, np.ndarray | float] | list[tuple[Configuration, np.ndarray | float]] | None:
        configs = [h[0] for h in self.history]
        costs = [h[1] for h in self.history]
        idx = np.argmin(costs)
        config = configs[idx]
        cost = costs[idx]
        return (config, cost)

