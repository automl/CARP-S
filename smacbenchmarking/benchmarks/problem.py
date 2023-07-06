from __future__ import annotations

from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo, TrialValue


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
    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # TODO Maybe add logger to track all interesting info here
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        TrialValue
            Value of the trial, i.e.:
                - cost : float | list[float]
                - time : float, defaults to 0.0
                - status : StatusType, defaults to StatusType.SUCCESS
                - starttime : float, defaults to 0.0
                - endtime : float, defaults to 0.0
                - additional_info : dict[str, Any], defaults to {}
        """
        raise NotImplementedError
