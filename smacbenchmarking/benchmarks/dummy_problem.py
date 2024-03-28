import time

from ConfigSpace import ConfigurationSpace, Float

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DummyProblem(Problem):
    def __init__(self, return_value=0, **kwargs):
        self._return_value = return_value
        self._configspace = ConfigurationSpace(
            space={
                "a": Float("a", bounds=(-1, 1)),
            }
        )
        super().__init__(**kwargs)

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        start_time = time.time()
        return TrialValue(cost=self._return_value, time=1, starttime=start_time, endtime=start_time + 1,virtual_time=1)
