"""ObjectiveFunction to optimize (base definition)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING

from ConfigSpace.util import deactivate_inactive_hyperparameters

from carps.utils.trials import TrialInfo

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.trials import TrialValue


class ObjectiveFunction(ABC):
    """ObjectiveFunction to optimize."""

    def __init__(self, loggers: list[AbstractLogger] | None = None) -> None:
        """Initialize ObjectiveFunction.

        Counts the number of function calls and trials.
        The number of trials can be fractional in the case for multi-fidelity optimization.

        Parameters
        ----------
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None
        """
        super().__init__()

        self.loggers: list[AbstractLogger] = loggers if loggers is not None else []
        self.n_trials: float = 0
        self.n_function_calls: int = 0

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
        """Evaluate objective function.

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

    def _make_config_valid(self, trial_info: TrialInfo) -> TrialInfo:
        """Make configuration valid.

        Some optimizers cannot handle conditional search spaces and thus propose configurations with values for
        inactive hyperparameters (HPs). Deactivate those to obtain a valid configuration
        for conditional search spaces.

        Args:
            trial_info (TrialInfo): Trial information.

        Returns:
            TrialInfo: Updated trial information with a valid configuration.
        """
        config: Configuration = trial_info.config
        config = deactivate_inactive_hyperparameters(configuration=dict(config), configuration_space=self.configspace)
        trial_info_dict = asdict(trial_info)
        trial_info_dict["config"] = config
        return TrialInfo(**trial_info_dict)

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate objective function.

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
        trial_info = self._make_config_valid(trial_info=trial_info)

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
