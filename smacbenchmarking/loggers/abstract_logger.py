from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


# TODO log optimizer's execution time

class AbstractLogger(ABC):
    def __init__(self) -> None:
        """AbstractLogger

        Intercepts the trial info and value during evaluate.
        """
        pass

    @abstractmethod
    def log_trial(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Log the trial.

        Parameters
        ----------
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        """
        raise NotImplementedError
