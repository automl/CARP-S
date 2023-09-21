from __future__ import annotations

from typing import Any

import time

from ConfigSpace import ConfigurationSpace
from hpobench.container.benchmarks.ml.lr_benchmark import LRBenchmark
from hpobench.container.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class HPOBenchProblem(Problem):
    def __init__(self, model: str, task_id: int, budget_type: str, seed: int):
        """Initialize a HPOBench problem.

        Parameters
        ----------
        model : str Model name.
        task_id : str Task ID, see https://arxiv.org/pdf/2109.06716.pdf, page 22.
        budget_type : str Budget type that is available for the model.
        seed: int Random seed.
        """
        super().__init__()

        self.budget_type = budget_type
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
        starttime = time.time()

        budget_value = float(trial_info.budget) if self.budget_type == "subsample" else int(trial_info.budget)

        result_dict = self._problem.objective_function(
            configuration=configuration, fidelity={self.budget_type: budget_value}, rng=trial_info.seed
        )
        endtime = time.time()
        T = endtime - starttime
        # function_value is 1 - accuracy on the validation set
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
        problem = LRBenchmark(
            rng=seed,
            task_id=task_id,
            container_name="lr_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "nn":
        problem = NNBenchmark(
            rng=seed,
            task_id=task_id,
            container_name="nn_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "rf":
        problem = RandomForestBenchmark(
            rng=seed,
            task_id=task_id,
            container_name="rf_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "svm":
        problem = SVMBenchmark(
            rng=seed,
            task_id=task_id,
            container_name="svm_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    elif model == "xgboost":
        problem = XGBoostBenchmark(
            rng=seed,
            task_id=task_id,
            container_name="xgboost_benchmark",
            container_source="./smacbenchmarking/singularity/containers/hpobench",
        )
    else:
        raise ValueError(f"Unknown model {model} for HPOBench.")

    return problem
