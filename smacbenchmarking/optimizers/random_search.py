from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace

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

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        return TrialInfo(config=config)

    def run(self) -> None:
        for i in range(self.n_trials):
            trial = self.problem.configspace.sample_configuration()
            trial = self.convert_to_trial(trial)
            _ = self.problem.evaluate(trial)
