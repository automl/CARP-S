from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem: Problem, n_trials: int) -> None:
        super().__init__(problem)

        self.configspace: ConfigurationSpace = self.problem.configspace
        self.n_trials: int = n_trials

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        return TrialInfo(config=config)
    
    def ask(self) -> TrialInfo:
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config=config)
    
    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        pass

    def _setup_optimizer(self) -> None:
        return None

