from __future__ import annotations

from abc import ABC, abstractmethod

from smacbenchmarking.utils.types import Incumbent
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class AbstractLogger(ABC):
    def __init__(self) -> None:
        """AbstractLogger

        Intercepts the trial info and value during evaluate.
        """
        pass

    @abstractmethod
    def log_trial(self, n_trials: int, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Log the trial.

        Parameters
        ----------
        n_trials : int
            The number of trials that have been run so far.
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        """
        raise NotImplementedError

    @abstractmethod
    def log_incumbent(self, incumbent: Incumbent) -> None:
        """Log the incumbents.

        Parameters
        ----------
        incumbent : Incumbent
            The incumbent (or multiple incumbents).
        """
        raise NotImplementedError

    @abstractmethod
    def log_arbitrary(self, data: dict, entity: str) -> None:
        """Log arbitrary data.

        Parameters
        ----------
        entity : str
            The entity to which to log (e.g. filename, table name).
        data : dict
            The data to log.
        """
        raise NotImplementedError
