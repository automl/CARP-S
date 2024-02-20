"""Yahpo problem class."""

from __future__ import annotations

from typing import Optional

import time

from ConfigSpace import ConfigurationSpace
from yahpo_gym import BenchmarkSet, list_scenarios, local_config

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class YahpoProblem(Problem):
    """Yahpo Problem."""

    def __init__(
        self, bench: str, instance: str, metric: str, budget_type: Optional[str] = None, lower_is_better: bool = True
    ):
        """Initialize a Yahpo problem.

        Parameters
        ----------
        bench: str Benchmark name.
        instance : str Instance name.
        metric : str Metric to optimize for (depends on the Benchmark instance e.g. lcbench).
        budget_type : Optional[str] Budget type for the multifidelity setting. Should be None for the blackbox setting.
        lower_is_better: bool Whether the metric is to be minimized or maximized.
        """
        super().__init__()

        assert bench in list_scenarios(), f"The scenario {bench} you choose is not available."

        # setting up meta data for surrogate benchmarks
        local_config.init_config()
        local_config.set_data_path("data/yahpo_data")

        self.scenario = bench
        self.instance = str(instance)

        self._problem = BenchmarkSet(scenario=bench, instance=self.instance)
        self._configspace = self._problem.get_opt_space(drop_fidelity_params=True)
        self.fidelity_space = self._problem.get_fidelity_space()
        self.fidelity_dims = list(self._problem.get_fidelity_space()._hyperparameters.keys())

        self.budget_type = budget_type
        self.lower_is_better = lower_is_better

        assert self.budget_type in self.fidelity_dims + [None], (
            f"The budget type {self.budget_type} you choose is "
            f"not available in this instance. Please choose "
            f"from {self.fidelity_dims + [None]}."
        )

        if self.budget_type is None or len(self.fidelity_dims) > 1:
            other_fidelities = [fid for fid in self.fidelity_dims if fid != self.budget_type]
            self.max_other_fidelities = {}
            for fidelity in other_fidelities:
                self.max_other_fidelities[fidelity] = self.fidelity_space.get_hyperparameter(fidelity).upper

        self.metric = metric

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    # @property
    # FIXME: see caro's message:
    #  the idea is somehow to overwrite the optimizer/multifidelity attributes for
    #  budget_variable and min_budget, max_budget with a FidelitiySpace class without interpolation
    #  that is based on the problem instance / config file. Similarily find out how to deal with
    #  the metrics.
    # def fidelity_space(self):
    #     return FidelitySpace(self.fidelity_dims)

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
        xs = configuration.get_dictionary()

        # If there are multiple fidelities, we take maximum fidelity value for the respective other fidelity dimensions
        # If we are in the blackbox setting, we take maximum fidelity value for all fidelity dimensions
        starttime = time.time()
        if self.budget_type is not None:
            if self.budget_type == "trainsize":
                xs.update({self.budget_type: trial_info.budget})
            else:
                if trial_info.budget is not None:  # to avoid mypy error
                    xs.update({self.budget_type: round(trial_info.budget)})

        if self.budget_type is None or len(self.fidelity_dims) > 1:
            xs.update(self.max_other_fidelities)

        # Benchmarking suite returns a list of results (as potentially more than one config can be passed),
        # as we only pass one config we need to select the first one
        if self.lower_is_better:
            cost = self._problem.objective_function(xs)[0][self.metric]
        else:
            cost = -self._problem.objective_function(xs)[0][self.metric]

        endtime = time.time()
        T = endtime - starttime

        trial_value = TrialValue(cost=float(cost), time=T, starttime=starttime, endtime=endtime)
        return trial_value
