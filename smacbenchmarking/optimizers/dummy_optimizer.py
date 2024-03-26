from __future__ import annotations

from typing import Optional, Any

from time import sleep

from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DummyOptimizer(Optimizer):

    def __init__(self, problem: Problem, dummy_cfg: DictConfig) -> None:
        super().__init__(problem)
        self.cfg = dummy_cfg
        self.trajectory = []
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
        trial = self.problem.configspace.sample_configuration()
        trial = self.convert_to_trial(trial, budget=self.budget)
        return trial

    def tell(self, trial_value: TrialValue) -> None:
        self.trajectory.append(float(trial_value.cost))



