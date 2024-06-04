from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING, Any

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from omegaconf import DictConfig

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent, SearchSpace


class DummyOptimizer(Optimizer):
    def __init__(
        self,
        dummy_cfg: DictConfig,
        problem: Problem,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)
        self.cfg = dummy_cfg
        self.history: list[TrialInfo, TrialValue] = []
        if "budget" in self.cfg:
            self.fidelity_enabled = True
            self.budget = self.cfg.budget
        else:
            self.fidelity_enabled = False
            self.budget = None

    def _setup_optimizer(self) -> Any:
        pass

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration, budget: float | None = None) -> TrialInfo:
        return TrialInfo(config=config, budget=budget)

    def ask(self) -> TrialInfo:
        sleep(1)
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config, budget=self.budget)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info, trial_value))

    def get_current_incumbent(self) -> Incumbent:
        return min(self.history, key=lambda x: x[1].cost)
