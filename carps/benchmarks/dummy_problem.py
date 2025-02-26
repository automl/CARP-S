"""Dummy problem for testing purposes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ConfigSpace import ConfigurationSpace, Float

from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger


class DummyProblem(Problem):
    """Dummy problem for testing purposes."""

    def __init__(
        self, return_value=0, budget_type: str | None = "dummy", loggers: list[AbstractLogger] | None = None
    ) -> None:
        """Initialize Dummy Problem.

        Parameters
        ----------
        return_value : float, optional
            Return value for the objective function evaluation, by default 0.
        budget_type : str | None, optional
            Budget type, by default "dummy"
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None
        """
        super().__init__(loggers)
        self.budget_type = budget_type
        self._return_value = return_value
        self._configspace = ConfigurationSpace(
            space={
                "a": Float("a", bounds=(-1, 1)),
            }
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        """Configuration Space."""
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:  # noqa: ARG002
        """Evaluate problem.

        Ignore trial_info, just return the return value.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial information.
        """
        start_time = time.time()
        return TrialValue(cost=self._return_value, time=1, starttime=start_time, endtime=start_time + 1, virtual_time=1)
