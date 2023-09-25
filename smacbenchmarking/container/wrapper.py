import requests
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json

from smacbenchmarking.benchmarks.loggingproblem import LoggingProblem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class ContainerizedProblemClient(LoggingProblem):
    def __init__(self):
        super().__init__()
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
