from __future__ import annotations

from typing import Any

import time

from ConfigSpace import ConfigurationSpace
from hpobench.container.benchmarks.ml.lr_benchmark import LRBenchmarkBB
from hpobench.container.benchmarks.ml.nn_benchmark import NNBenchmarkBB
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmarkBB
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class HPOBenchProblem(Problem):
    def __init__(self, model: str, task_id: int, seed: int):
        super().__init__()

        self._problem = get_hpobench_problem(task_id=task_id, model=model, seed=seed)
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
        trial_value = TrialValue(cost=result_dict["function_value"], time=T, starttime=starttime, endtime=endtime)
        return trial_value


def get_hpobench_problem(model: str, task_id: int, seed: int) -> Any:
    r"""Get HPOBench problem

    Parameters
    ----------
    model : str
    task_id : int
    seed : int
        Seed for configuration space.

    Returns
    -------
    Any
        Target function.
    """
    if model == "lr":
        problem = LRBenchmarkBB(
            rng=seed,
            task_id=task_id,
            container_name="lr_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "nn":
        problem = NNBenchmarkBB(
            rng=seed,
            task_id=task_id,
            container_name="nn_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "rf":
        problem = RandomForestBenchmarkBB(
            rng=seed,
            task_id=task_id,
            container_name="rf_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "svm":
        problem = SVMBenchmarkBB(
            rng=seed,
            task_id=task_id,
            container_name="svm_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "xgboost":
        problem = XGBoostBenchmarkBB(
            rng=seed,
            task_id=task_id,
            container_name="xgboost_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    else:
        raise ValueError(f"Unknown model {model} for HPOBench.")

    return problem
