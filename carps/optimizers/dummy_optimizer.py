from __future__ import annotations

from typing import Optional, Any
import numpy as np

from time import sleep

from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig

from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer, SearchSpace
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Cost


class DummyOptimizer(Optimizer):

    def __init__(
            self,
            dummy_cfg: DictConfig,
            problem: Problem,
            n_trials: int | None,
            time_budget: float | None,
            n_workers: int = 1,
            loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, n_trials, time_budget, n_workers, loggers)
        self.cfg = dummy_cfg
        self.history: list[Configuration, Cost] = []
        if "budget" in self.cfg.keys():
            self.fidelity_enabled = True
            self.budget = self.cfg.budget
        else:
            self.fidelity_enabled = False
            self.budget = None

    def _setup_optimizer(self) -> Any:
        pass

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration, budget: Optional[float] = None) -> TrialInfo:
        return TrialInfo(config=config, budget=budget)

    def ask(self) -> TrialInfo:
        sleep(1)
        config = self.problem.configspace.sample_configuration()
        trial = self.convert_to_trial(config, budget=self.budget)
        return trial

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info.config, trial_value.cost))

    def get_current_incumbent(self) -> tuple[Configuration, np.ndarray | float] | list[tuple[Configuration, np.ndarray | float]] | None:
        return self.history[np.array([v[1] for v in self.history]).argmin()]



