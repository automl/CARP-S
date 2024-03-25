from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ConfigSpace import ConfigurationSpace

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo

SearchSpace = Any


class Optimizer(ABC):
    def __init__(self, problem: Problem, n_trials: int | None, time_budget: float | None) -> None:
        self.problem = problem
        if n_trials is None and time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget` "
                             "as the optimization budget.")
        self.n_trials: int | None = n_trials
        self.time_budget: float | None = time_budget
        
        super().__init__()
        # This indicates if the optimizer can deal with multi-fidelity optimization
        self.fidelity_enabled = False

    @abstractmethod
    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        """Convert ConfigSpace configuration space to search space from optimizer.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns
        -------
        SearchSpace
            Optimizer's search space.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Run Optimizer on Problem"""
        raise NotImplementedError
