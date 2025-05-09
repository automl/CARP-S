"""HPOBench objective function class."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from hpobench.container.benchmarks.ml.lr_benchmark import LRBenchmark
from hpobench.container.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from omegaconf import ListConfig

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from hpobench.abstract_benchmark import AbstractBenchmark
    from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

    from carps.loggers.abstract_logger import AbstractLogger


class HPOBenchObjectiveFunction(ObjectiveFunction):
    """HPOBench objective function."""

    def __init__(
        self,
        seed: int,
        model: str | None = None,
        task_id: int | None = None,
        metric: str | list[str] = "function_value",
        benchmark_client: AbstractBenchmarkClient | None = None,
        budget_type: str | None = None,
        loggers: list[AbstractLogger] | None = None,
    ):
        """Initialize a HPOBench objective function.

        Either specify model and task_id for an ML tabular benchmark objective function or benchmark_client.

        Parameters
        ----------
        model : str | None, Model name.
        task_id : str Task ID, see https://arxiv.org/pdf/2109.06716.pdf, page 22.
        metric : str | list[str], Metric to optimize.
        benchmark_client : AbstractBenchmarkClient
            Instantiated benchmark/objective function, e.g.
            `hpobench.container.benchmarks.surrogates.paramnet_benchmark.ParamNetAdultOnStepsBenchmark`.
        seed: int Random seed.
        budget_type : Optional[str] Budget type for the multifidelity setting.
                      Should be None for the blackbox setting.
        """
        super().__init__(loggers)

        self.budget_type = budget_type

        if benchmark_client is None and model is None and task_id is None:
            raise ValueError("Please specify either benchmark_client or model and task_id.")
        if benchmark_client is None:
            assert model is not None
            assert task_id is not None
            objective_function = get_tabular_objective_function(
                task_id=task_id, model=model, seed=seed, budget_type=self.budget_type
            )
        else:
            objective_function = benchmark_client
        self._benchmark: AbstractBenchmarkClient | AbstractBenchmark = objective_function
        if not isinstance(metric, list | ListConfig):
            metric = [metric]
        self.metric = metric
        self._configspace = self._benchmark.get_configuration_space(seed=seed)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate objective function.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        TrialValue
            Cost
        """
        configuration = trial_info.config
        starttime = time.time()

        if trial_info.budget is not None and self.budget_type is not None:
            budget_value = float(trial_info.budget) if self.budget_type == "subsample" else round(trial_info.budget)
            fidelity = {self.budget_type: budget_value}
        else:
            fidelity = None

        result_dict = self._benchmark.objective_function(
            configuration=configuration, fidelity=fidelity, rng=trial_info.seed
        )
        endtime = time.time()
        T = endtime - starttime
        virtual_time = result_dict["cost"] if trial_info.budget is None else 0.0
        costs = [result_dict[metric] for metric in self.metric]
        if len(costs) == 1:
            costs = costs[0]
        # function_value is 1 - accuracy on the validation set
        return TrialValue(
            cost=costs,
            time=T,
            starttime=starttime,
            endtime=endtime,
            virtual_time=virtual_time,
        )


def get_tabular_objective_function(
    model: str, task_id: int, seed: int, budget_type: str | None = None
) -> TabularBenchmark:
    """Get HPOBench objective function / TabularBenchmark.

    Parameters
    ----------
        model : str Model name.
        task_id : str Task ID, see https://arxiv.org/pdf/2109.06716.pdf, page 22.
        seed: int Random seed.
        budget_type : Optional[str] Budget type for the multifidelity setting.
                      Should be None for the blackbox setting.

    Returns:
    -------
    TabularBenchmark
        Ojbective function.
    """
    common_args = {"rng": seed, "task_id": task_id}
    if model == "lr":
        objective_function = (
            TabularBenchmark(model="lr", **common_args) if budget_type is None else LRBenchmark(**common_args)
        )
    elif model == "nn":
        objective_function = (
            TabularBenchmark(model="nn", **common_args) if budget_type is None else NNBenchmark(**common_args)
        )
    elif model == "rf":
        objective_function = (
            TabularBenchmark(model="rf", **common_args) if budget_type is None else RandomForestBenchmark(**common_args)
        )
    elif model == "svm":
        objective_function = (
            TabularBenchmark(model="svm", **common_args) if budget_type is None else SVMBenchmark(**common_args)
        )
    elif model == "xgboost":
        objective_function = (
            TabularBenchmark(model="xgb", **common_args) if budget_type is None else XGBoostBenchmark(**common_args)
        )
    else:
        raise ValueError(f"Unknown model {model} for HPOBench.")

    return objective_function
