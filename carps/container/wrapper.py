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
    def __init__(self, n_workers: int = 1, loggers: list[AbstractLogger] | None = None):
        super().__init__(loggers=loggers)
        self.n_workers = n_workers
        self._configspace = None

    @property
    def configspace(self) -> ConfigurationSpace:
        if self._configspace is None:
            # ask server about configspace
            response = requests.get("http://localhost:5000/configspace")
            self._configspace = cs_json.read(response.json())

        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # ask server about evaluation
        response = requests.post("http://localhost:5000/evaluate", json=trial_info.to_json())
        return TrialValue.from_json(response.json())

    def f_min(self) -> float | None:
        raise NotImplementedError("f_min is not yet implemented for ContainerizedProblemClient")
