import requests
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
from flask import app, request

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class ContainerizedProblemClient(Problem):
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

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # ask server about evaluation
        response = requests.post("http://localhost:5000/evaluate", json=trial_info.to_json())
        print(response)
        print(type(response))
        print(response.json())
        print(type(response.json()))
        return TrialValue.from_json(response.json())


class ContainerizedProblemServer(Problem):
    def __init__(self, problem: Problem):
        super().__init__()
        self._problem = problem

    # @app.route("/configspace", methods=["GET"])
    def _request_configspace(self):
        return cs_json.write(self.configspace)

    # @app.route("/evaluate", methods=["POST"])
    def _request_evaluation(self):
        if request.is_json:
            print(request.get_json())
            trial_info = TrialInfo(**request.get_json())
            return self.evaluate(trial_info).to_json()
        else:
            raise ValueError("Request is not JSON.")

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._problem.configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self._problem.evaluate(trial_info)
