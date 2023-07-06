from __future__ import annotations

import time

from ConfigSpace import ConfigurationSpace

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

from hpobench.container.benchmarks.ml.lr_benchmark import LRBenchmarkBB
from hpobench.container.benchmarks.ml.nn_benchmark import NNBenchmarkBB
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmarkBB
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB


class HPOBenchProblem(Problem):
    def __init__(self, model: str, task_id: int, seed: int):
        super().__init__()
        self.model = model
        self.task_id = task_id

        if model == "lr":
            self._problem = LRBenchmarkBB(rng=seed, task_id=task_id)
        elif model == "nn":
            self._problem = NNBenchmarkBB(rng=seed, task_id=task_id)
        elif model == "rf":
            self._problem = RandomForestBenchmarkBB(rng=seed, task_id=task_id)
        elif model == "svm":
            self._problem = SVMBenchmarkBB(rng=seed, task_id=task_id)
        elif model == "xgboost":
            self._problem = XGBoostBenchmarkBB(rng=seed, task_id=task_id, container_name="xgboost_benchmark",
                                   container_source='./smacbenchmarking/singularity/containers/hpobench')
        else:
            raise ValueError(f"Unknown model {model} for HPOBench.")

        self._configspace = self._problem.get_configuration_space(seed=seed)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        TrialValue
            Cost
        """
        configuration = trial_info.config
        seed = trial_info.seed
        starttime = time.time()
        result_dict = self._problem.objective_function(configuration=configuration, rng=seed)
        endtime = time.time()
        T = endtime - starttime
        trial_value = TrialValue(cost=result_dict['function_value'], time=T, starttime=starttime, endtime=endtime)
        return trial_value
