"""Yahpo problem class."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import ListConfig
from yahpo_gym import BenchmarkSet, list_scenarios, local_config

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger

LOWER_IS_BETTER = {
    "mmce": True,  # classification error
    "f1": False,
    "auc": False,
    "logloss": True,
    "nf": True,  # number of features used
    "ias": True,  # interaction strength of features  # TODO check
    "rammodel": True,  # model size
    "val_accuracy": False,
    "val_cross_entropy": True,
    "acc": False,
    "bac": False,  # balanced acc
    "brier": True,
    "memory": True,
    "timetrain": True,
    "runtime": True,
    "mec": True,  # main effect complexity of features
    "valid_mse": True,
}


def maybe_invert(value: float, target: str) -> float:
    """Maybe invert the objective.

    E.g., in carps we minimize, but accuracy needs to be maximized.

    Parameters
    ----------
    value : float
        Value to maybe invert.
    target : str
        Target metric.

    Returns:
    -------
    float
        Maybe negated objective function value.
    """
    sign = 1
    if not LOWER_IS_BETTER[target]:
        sign = -1
    return sign * value


class YahpoObjectiveFunction(ObjectiveFunction):
    """Yahpo ObjectiveFunction."""

    def __init__(
        self,
        bench: str,
        instance: str,
        metric: str | list[str],
        budget_type: str | None = None,
        yahpo_data_path: str | None = None,
        seed: int = 0,
        loggers: list[AbstractLogger] | None = None,
    ):
        """Initialize a Yahpo problem.

        Parameters
        ----------
        bench: str
            Benchmark name.
        instance : str
            Instance name.
        metric : str
            Metric(s) to optimize for (depends on the Benchmark instance e.g. lcbench).
        budget_type : Optional[str]
            Budget type for the multifidelity setting. Should be None for the blackbox setting.
        yahpo_data_path : str | None
            Path to yahpo data, defaults to '../benchmark_data/yahpo_data' (relative to this file).
        seed : int
            Seed for configuration space.
        loggers : list[AbstractLogger] | None
            List of loggers to log information.
        """
        super().__init__(loggers)

        assert bench in list_scenarios(), f"The scenario {bench} you choose is not available."

        yahpo_data_path_path = yahpo_data_path or Path(__file__).parent.parent / "benchmark_data/yahpo_data"

        # setting up meta data for surrogate benchmarks
        local_config.init_config()
        local_config.set_data_path(yahpo_data_path_path)

        self.scenario = bench
        self.instance = str(instance)

        # No multithread in YAHPO Benchmark because this leads to this error:
        # Traceback (most recent call last):
        #     File "/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carpsenv/lib/python3.12/site-packages/hydra/_internal/instantiate/_instantiate2.py", line 92, in _call_target  # noqa: E501
        #         return _target_(*args, **kwargs)
        #             ^^^^^^^^^^^^^^^^^^^^^^^^^
        #     File "/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/objective_functions/yahpo.py", line 104, in __init__  # noqa: E501
        #         local_config.set_data_path(yahpo_data_path_path)
        #     File "/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/lib/yahpo_gym/yahpo_gym/yahpo_gym/local_config.py", line 59, in set_data_path  # noqa: E501
        #         config.update({'data_path': str(data_path)})
        #         ^^^^^^^^^^^^^
        #     AttributeError: 'NoneType' object has no attribute 'update'
        # Which occurs when having several instances of the same benchmark
        self._problem = BenchmarkSet(scenario=bench, instance=self.instance, check=False, multithread=False)
        self._configspace = self._problem.get_opt_space(drop_fidelity_params=True, seed=seed)
        self.fidelity_space = self._problem.get_fidelity_space()
        self.fidelity_dims = list(self._problem.get_fidelity_space().keys())

        self.budget_type = budget_type

        assert self.budget_type in [*self.fidelity_dims, None], (
            f"The budget type {self.budget_type} you choose is "
            f"not available in this instance. Please choose "
            f"from {[*self.fidelity_dims, None]}."
        )

        if self.budget_type is None or len(self.fidelity_dims) > 1:
            other_fidelities = [fid for fid in self.fidelity_dims if fid != self.budget_type]
            self.max_other_fidelities = {}
            for fidelity in other_fidelities:
                self.max_other_fidelities[fidelity] = self.fidelity_space[fidelity].upper

        if not isinstance(metric, list | ListConfig):
            metric = [metric]
        self.metrics = metric

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    # @property
    # FIXME: see caro's message:
    #  the idea is somehow to overwrite the optimizer/multifidelity attributes for
    #  budget_variable and min_fidelity, max_fidelity with a FidelitiySpace class without interpolation
    #  that is based on the problem instance / config file. Similarly find out how to deal with
    #  the metrics.
    # def fidelity_space(self):
    #     return FidelitySpace(self.fidelity_dims)

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

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
        xs = dict(configuration)

        # If there are multiple fidelities, we take maximum fidelity value for the respective other fidelity dimensions
        # If we are in the blackbox setting, we take maximum fidelity value for all fidelity dimensions
        starttime = time.time()
        if self.budget_type is not None:
            if self.budget_type == "trainsize":
                xs.update({self.budget_type: trial_info.budget})
            elif trial_info.budget is not None:  # to avoid mypy error
                xs.update({self.budget_type: round(trial_info.budget)})

        if self.budget_type is None or len(self.fidelity_dims) > 1:
            xs.update(self.max_other_fidelities)

        # Benchmarking suite returns a list of results (as potentially more than one config can be passed),
        # as we only pass one config we need to select the first one
        ret = self._problem.objective_function(configuration=xs, seed=trial_info.seed)[0]
        costs = [maybe_invert(ret[target], target) for target in self.metrics]
        virtual_time = ret.get("time", 0.0)
        if len(costs) == 1:
            costs = costs[0]  # type: ignore[assignment]

        endtime = time.time()
        T = endtime - starttime

        return TrialValue(cost=costs, time=T, starttime=starttime, endtime=endtime, virtual_time=virtual_time)
