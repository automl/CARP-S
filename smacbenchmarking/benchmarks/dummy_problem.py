from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DummyProblem(Problem):
    def __init__(self, return_value=0, lower=0, upper=1):
        super().__init__()
        self._return_value = return_value
        self._configspace = ConfigurationSpace()
        self._configspace.add_hyperparameter(
            UniformFloatHyperparameter("x", lower=lower, upper=upper)
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return TrialValue(cost=self._return_value, time=0, starttime=0, endtime=0)
