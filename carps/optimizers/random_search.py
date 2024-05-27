from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer, SearchSpace
from carps.utils.task import Task
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent


class RandomSearchOptimizer(Optimizer):
    def __init__(
            self,
            problem: Problem,
            task: Task,
            loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)

        self.configspace: ConfigurationSpace = self.problem.configspace
        self.history: list[tuple[TrialInfo, TrialValue]] = []
        self.is_multifidelity = task.is_multifidelity

        if hasattr(task, "n_objectives"):
            self.is_multiobjective = task.n_objectives > 1

        if self.is_multiobjective:
            # reduce import dependency to pymoo, if not MO
            # from pymoo.indicators.hv import Hypervolume
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

            # self.HV = Hypervolume
            self.NDS = NonDominatedSorting

            # if hasattr(task, 'ref_point'):
            #     # give user the option to specify the reference point (if known in advance)
            #     self.ref_point = task.ref_point
            #     self.hv = self.HV(ref_point=np.array(task.ref_point))

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        budget = None
        if self.is_multifidelity:
            budget = self.task.max_budget
        return TrialInfo(config=config, budget=budget)

    def ask(self) -> TrialInfo:
        config = self.problem.configspace.sample_configuration()
        return self.convert_to_trial(config=config)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info, trial_value))

    def _setup_optimizer(self) -> None:
        return None

    def get_current_incumbent(self) -> Incumbent:
        if self.is_multiobjective:
            max_budget = np.max([v[0].budget for v in self.history])
            highest_fidelity = [v for v in self.history if v[0].budget == max_budget]
            hf_cost = np.array([v[1].cost for v in highest_fidelity])

            # # calculate the hypervolume on the highest fidelity!
            # if not hasattr(self, 'ref_point'):
            #     # calculate the reference point as relative margin of the highest fidelity points
            #     ref_point = hf_cost.max(axis=0)
            #     self.hv = self.HV(ref_point=ref_point)
            # hv = self.hv(hf_cost)

            non_dom = self.NDS().do(hf_cost, only_non_dominated_front=True)

            # enforce an order on the non-dominated solutions, to avoid bad comparisons
            return list(
                sorted([self.history[i] for i in non_dom], key=lambda x: x[1].cost)
            )
        else:
            return min(self.history, key=lambda x: x[1].cost)
