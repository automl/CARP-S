from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.trials import TrialInfo, TrialValue


class Problem(ABC):
    """Problem to optimize."""

    def __init__(self, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__()

        self.loggers: list[AbstractLogger] = loggers if loggers is not None else []
        self.n_trials: float = 0
        self.n_function_calls: int = 0.0

    @property
    def f_min(self) -> float | None:
        """Return the minimum function value.

        Returns:
        -------
        float | None
            Minimum function value (if exists).
            Else, return None.
        """
        return None

    @property
    @abstractmethod
    def configspace(self) -> ConfigurationSpace:
        """Configuration Space.

        All optimizers need to receive a configspace and
        convert it to their search space definition.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance, name, checkpoint.

        Returns:
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

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        trial_value = self._evaluate(trial_info=trial_info)
        self.n_function_calls += 1
        if trial_info.normalized_budget is not None:
            self.n_trials += trial_info.normalized_budget
        else:
            self.n_trials += 1

        for logger in self.loggers:
            logger.log_trial(
                n_trials=self.n_trials,
                n_function_calls=self.n_function_calls,
                trial_info=trial_info,
                trial_value=trial_value,
            )

        return trial_value
