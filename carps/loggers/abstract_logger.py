from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carps.utils.trials import TrialInfo, TrialValue
    from carps.utils.types import Incumbent


class AbstractLogger(ABC):
    def __init__(self) -> None:
        """AbstractLogger.

        Intercepts the trial info and value during evaluate.
        """

    @abstractmethod
    def log_trial(
        self, n_trials: float, trial_info: TrialInfo, trial_value: TrialValue, n_function_calls: int | None = None
    ) -> None:
        """Log the trial.

        Parameters
        ----------
        n_trials : float
            The number of trials that have been run so far.
            For the case of multi-fidelity, a full trial
            is a configuration evaluated on the maximum budget and
            the counter is increased by `budget/max_budget` instead
            of 1.
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        n_function_calls: int | None, default None
            The number of target function calls, no matter the budget.
        """
        raise NotImplementedError

    @abstractmethod
    def log_incumbent(self, n_trials: int, incumbent: Incumbent) -> None:
        """Log the incumbents.

        Parameters
        ----------
        n_trials : int
            The number of trials that have been run so far.
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
