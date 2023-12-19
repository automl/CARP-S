from __future__ import annotations

from time import sleep, time

from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace
from smacbenchmarking.utils.trials import TrialInfo


class DummyOptimizer(Optimizer):
    def __init__(self, problem: Problem, dummy_cfg: DictConfig) -> None:
        super().__init__(problem)
        self.cfg = dummy_cfg
        self.trajectory = []
        if self.cfg.budget is not None:
            self.fidelity_enabled = True

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration, budget: float | None = None) -> TrialInfo:
        return TrialInfo(config=config, budget=budget)

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        return list(range(len(self.trajectory))), self.trajectory

    def run(self) -> None:
        timeout = self.cfg.timeout
        budget = self.cfg.budget
        start_time = time()
        while time() - start_time < timeout:
            sleep(1)
            trial = self.problem.configspace.sample_configuration()
            trial = self.convert_to_trial(trial, budget=budget)
            result = self.problem.evaluate(trial)
            self.trajectory.append(float(result.cost))
