from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo

from smacbenchmarking.benchmarks.problem import Problem

SearchSpace = Any


class Optimizer(ABC):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
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
    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        # TODO return a list of tuples. Something more verbose like a dict or DataFrame would be better.
        """List of x and y values of the incumbents over time. x depends on ``sort_by``.

        Parameters
        ----------
        sort_by: str
            Can be "trials" or "walltime".

        Returns
        -------
        tuple[list[float], list[float]]

        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Run Optimizer on Problem"""
        raise NotImplementedError
