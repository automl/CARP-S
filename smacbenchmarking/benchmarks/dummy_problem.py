from ConfigSpace import ConfigurationSpace

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DummyProblem(Problem):
    def __init__(self, return_value=0):
        self._return_value = return_value
        self._configspace = ConfigurationSpace()

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return TrialValue(cost=self._return_value, time=0, starttime=0, endtime=0)