"""Dummy objective function for testing purposes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ConfigSpace import ConfigurationSpace, Float
from omegaconf import ListConfig

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger


class DummyObjectiveFunction(ObjectiveFunction):
    """Dummy objective function for testing purposes."""

    def __init__(
        self,
        return_value: float | list[float] = 0,
        configuration_space: ConfigurationSpace | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        """Initialize Dummy ObjectiveFunction.

        Parameters
        ----------
        return_value : float, optional
            Return value for the objective function evaluation, by default 0.
        configuration_space : ConfigurationSpace, optional
            Configuration space, by default None. Defaults to a space with one float parameter "a" in the range [-1, 1].
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None
        """
        super().__init__(loggers)
        self._return_value: float | list[float] = return_value
        if isinstance(self._return_value, int):
            self._return_value = float(self._return_value)
        if isinstance(self._return_value, ListConfig):
            self._return_value = list(self._return_value)
        assert isinstance(
            self._return_value, float | list
        ), f"Return value must be a float or a list of floats but is {type(self._return_value)}. {self._return_value}."
        self._configspace = configuration_space or ConfigurationSpace(
            space={
                "a": Float("a", bounds=(-1, 1)),
            }
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        """Configuration Space."""
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:  # noqa: ARG002
        """Evaluate objective function.

        Ignore trial_info, just return the return value.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial information.
        """
        start_time = time.time()
        return TrialValue(cost=self._return_value, time=1, starttime=start_time, endtime=start_time + 1, virtual_time=1)
