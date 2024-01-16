from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace
from smacbenchmarking.utils.trials import TrialInfo


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem: Problem, n_trials: int) -> None:
        super().__init__(problem)

        self.configspace: ConfigurationSpace = self.problem.configspace
        self.n_trials: int = n_trials

        self.trajectory_X: list[Configuration] = []
        self.trajectory_y: list[float] = []

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:
        return TrialInfo(config=config)

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        return (self.trajectory_X, self.trajectory_y)

    def run(self) -> None:
        best_y = 1e10
        for i in range(self.n_trials):
            trial = self.problem.configspace.sample_configuration()
            trial = self.convert_to_trial(trial)
            result = self.problem.evaluate(trial)
            if result.cost < best_y:
                best_y = result.cost
                self.trajectory_X.append(list(trial.config.get_dictionary().values()))
                self.trajectory_y.append(result.cost)
