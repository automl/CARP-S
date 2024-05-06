from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace

from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer, SearchSpace
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent
from carps.utils.task import Task


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

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        return TrialInfo(config=config)
    
    def ask(self) -> TrialInfo:
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config=config)
    
    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info, trial_value))

    def _setup_optimizer(self) -> None:
        return None
    
    def get_current_incumbent(self) -> Incumbent:
        return min(self.history, key=lambda x: x[1].cost)
