"""Wrapper for containerized problems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import requests
from ConfigSpace.read_and_write import json as cs_json

from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger


class ContainerizedProblemClient(Problem):
    """Wrapper for containerized problems."""

    def __init__(self, n_workers: int = 1, loggers: list[AbstractLogger] | None = None):
        """Initialize ContainerizedProblemClient.

        Parameters
        ----------
        n_workers : int, default 1
            Number of workers.
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None
        """
        super().__init__(loggers=loggers)
        self.n_workers = n_workers
        self._configspace = None

    @property
    def configspace(self) -> ConfigurationSpace:
        """Get the configuration space of the problem.

        Returns:
        -------
        ConfigurationSpace
            Configuration space of the problem.
        """
        if self._configspace is None:
            # ask server about configspace
            response = requests.get("http://localhost:5000/configspace")  # noqa: S113
            self._configspace = cs_json.read(response.json())

        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # ask server about evaluation
        response = requests.post("http://localhost:5000/evaluate", json=trial_info.to_json())  # noqa: S113
        return TrialValue.from_json(response.json())

    def f_min(self) -> float | None:
        """Get the minimum value of the objective function.

        Reises
        ------
        NotImplementedError
            f_min is not yet implemented for ContainerizedProblemClient
        """
        raise NotImplementedError("f_min is not yet implemented for ContainerizedProblemClient")
