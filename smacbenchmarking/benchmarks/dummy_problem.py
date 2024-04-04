import time

from ConfigSpace import ConfigurationSpace, Float

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DummyProblem(Problem):
    def __init__(
            self,
            return_value=0,
            budget_type: str | None = "dummy",
            loggers: list[AbstractLogger] | None = None
    ):
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
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        start_time = time.time()
        return TrialValue(cost=self._return_value, time=1, starttime=start_time, endtime=start_time + 1,virtual_time=1)
