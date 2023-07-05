from __future__ import annotations

from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo


class Problem(ABC):
    """Problem to optimize."""

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def configspace(self) -> ConfigurationSpace:
        """Configuration Space

        All optimizers need to receive a configspace and
        convert it to their search space definition.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, trial_info: TrialInfo) -> list[float] | float:
        # TODO Maybe add logger to track all interesting info here
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        list[float] | float
            Cost (vector).
        """
        raise NotImplementedError


class SingleObjectiveProblem(Problem):
    @abstractmethod
    def evaluate(self, trial_info: TrialInfo) -> float:  # TODO Return runtime of one eval as well?
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        float
            Cost
        """
        raise NotImplementedError


class MultiObjectiveProblem(Problem):
    @abstractmethod
    def evaluate(self, trial_info: TrialInfo) -> list[float]:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        list[float]
            Cost vector.
        """
        raise NotImplementedError
