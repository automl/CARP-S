from __future__ import annotations

from typing import Any, Callable

import copy
import datetime
import time
from collections import OrderedDict

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from syne_tune.backend.trial_status import Trial as SyneTrial
from syne_tune.config_space import (
    choice,
    lograndint,
    loguniform,
    ordinal,
    randint,
    uniform,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    BOHB,
    BORE,
    DEHB,
    KDE,
    MOBSTER,
    BayesianOptimization,
)
from syne_tune.optimizer.scheduler import TrialScheduler as SyneTrialScheduler

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

# This is a subset from the syne-tune baselines
optimizers_dict = {
    "BayesianOptimization": BayesianOptimization,
    # "ASHA": ASHA,
    "MOBSTER": MOBSTER,
    "BOHB": BOHB,
    "KDE": KDE,
    "BORE": BORE,
    # "DEHB": DEHB,
}

mf_optimizer_dicts = {"with_mf": {"ASHA", "DEHB", "MOBSTER"}, "without_mf": {"BORE", "BayesianOptimization", "KDE"}}


def configspaceHP2syneTuneHP(hp: Hyperparameter) -> Callable:
    if isinstance(hp, IntegerHyperparameter):
        if hp.log:
            return lograndint(hp.lower, hp.upper)
        else:
            return randint(hp.lower, hp.upper)
    elif isinstance(hp, FloatHyperparameter):
        if hp.log:
            return loguniform(hp.lower, hp.upper)
        else:
            return uniform(hp.lower, hp.upper)
    elif isinstance(hp, CategoricalHyperparameter):
        return choice(hp.choices)
    elif isinstance(hp, OrdinalHyperparameter):
        return ordinal(hp.sequence)
    elif isinstance(hp, Constant):
        return choice([hp.value])
    else:
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


class SynetuneOptimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        optimizer_name: "str",
        max_budget: float | None = None,
        num_trials: int | None = None,
        wallclock_times: float | None = None,
    ) -> None:
        super().__init__(problem)
        self.fidelity_enabled = False

        self.configspace = self.problem.configspace
        assert optimizer_name in optimizers_dict
        if optimizer_name in mf_optimizer_dicts["with_mf"]:
            raise NotImplementedError("Multi-Fidelity Optimization on SyneTune is not implemented yet!")
            self.fidelity_enabled = True
            if not hasattr(problem, "budget_type"):
                raise ValueError("To run multi-fidelity optimizer, the problem must have a budget_type!")
            if max_budget is None:
                raise ValueError("To run multi-fidelity optimizer, we must specify max_budget!")

        self.syne_tune_configspace = self.convert_configspace(self.configspace)
        self.metric = getattr(problem, "metric", "cost")
        self.budget_type = getattr(self.problem, "budget_type", None)
        self.trial_counter = 0
        self.max_budget = max_budget

        self.optimizer_name = optimizer_name
        self._optimizer: SyneTrialScheduler | None = None

        if num_trials is None and wallclock_times is None:
            raise ValueError("either num_trials or wallclock_times must be given!")
        self.max_num_trials = num_trials
        self.start_time = time.time()
        self.wallclock_times = wallclock_times

        self.completed_experiments: OrderedDict[int, tuple[TrialValue, TrialInfo]] = OrderedDict()

    def convert_configspace(self, configspace: ConfigurationSpace) -> dict[str, Any]:
        """Convert configuration space from Problem to Optimizer.

        Convert the configspace from ConfigSpace to syne-tune. However, given that syne-tune does not support
        conditions and forbidden clauses, we only add hyperparameters here

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        dict[str, Any]
        -------
        configspace_st
            Configuration space for syne tune.
        """
        configspace_st = {}
        for k, v in configspace.items():
            configspace_st[k] = configspaceHP2syneTuneHP(v)
        return configspace_st

    def convert_to_trial(  # type: ignore[override]
        self, config: Configuration, seed: int | None = None, budget: float | None = None, instance: str | None = None
    ) -> TrialInfo:
        """Convert proposal from SMAC to TrialInfo.

        Parameters
        ----------
        config : Configuration
            Configuration
        seed : int | None, optional
            Seed, by default None
        budget : float | None, optional
            Budget, by default None
        instance : str | None, optional
            Instance, by default None

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        trial_info = TrialInfo(config=config, seed=seed, budget=budget, instance=instance)
        return trial_info

    def ask(self) -> SyneTrial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self._optimizer.suggest(self.trial_counter)
        trial = SyneTrial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        return trial

    def evaluate(self, trial: SyneTrial) -> float:
        configs = copy.deepcopy(trial.config)
        if self.budget_type is not None:
            budget = configs.pop(self.budget_type)
        else:
            budget = None
        configuration = Configuration(configuration_space=self.configspace, values=configs)
        cost = self.target_function(config=configuration, budget=budget)
        return cost

    def tell(self, trial: SyneTrial, cost: float):
        """
        Feed experiment results back to the Scheduler

        :param trial: SyneTrial that was run
        :param cost: float, cost values
        """
        experiment_result = {self.metric: cost}
        self.trial_counter += 1
        self._optimizer.on_trial_complete(trial=trial, result=experiment_result)

    def target_function(
        self, config: Configuration, seed: int | None = None, budget: float | None = None, instance: str | None = None
    ) -> float:
        """Target Function

        Interface for the Problem.

        Parameters
        ----------
        config : Configuration
            Configuration
        seed : int | None, optional
            Seed, by default None
        budget : float | None, optional
            Budget, by default None
        instance : str | None, optional
            Instance, by default None

        Returns
        -------
        float
            cost
        """
        trial_info = self.convert_to_trial(config=config, seed=seed, budget=budget, instance=instance)
        trial_value = self.problem.evaluate(trial_info=trial_info)
        if self.wallclock_times is not None:
            if trial_value.endtime - self.start_time > self.wallclock_times:
                # In this case, it is actually timed out. We will simply ignore that
                return trial_value.cost
        self.completed_experiments[self.trial_counter] = (trial_value, trial_info)
        return trial_value.cost

    def setup_optimizer(self) -> SyneTrialScheduler:
        """
        Setup Optimizer.

        Retrieve defaults and instantiate SMAC.

        Returns
        -------
        SMAC4AC
            Instance of a SMAC facade.

        """
        optimizer_kwargs = dict(
            config_space=self.syne_tune_configspace,
            metric=getattr(self.problem, "metric", "cost"),
            mode="min",
        )
        if self.optimizer_name in mf_optimizer_dicts["with_mf"]:
            optimizer_kwargs["resource_attr"] = self.problem.budget_type
            optimizer_kwargs["max_t"] = self.max_budget

        bscheduler = optimizers_dict[self.optimizer_name](**optimizer_kwargs)
        return bscheduler

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        """List of x and y values of the incumbents over time. x depends on ``sort_by``.

        Parameters
        ----------
        sort_by: str
            Can be "trials" or "walltime".

        Returns
        -------
        tuple[list[float], list[float]]

        """
        # if len(self.task.objectives) > 1:
        #     raise NotSupportedError

        X: list[int | float] = []
        Y: list[float] = []

        current_incumbent = np.inf

        for k, v in self.completed_experiments.items():
            trial_value, trial_info = v
            cost = trial_value.cost
            if cost > 1e6:
                continue
            if current_incumbent < cost:
                current_incumbent = cost

                if sort_by == "trials":
                    X.append(k)
                elif sort_by == "walltime":
                    X.append(trial_value.endtime)
                else:
                    raise RuntimeError("Unknown sort_by.")

                Y.append(cost)

        return X, Y

    def run(self) -> None:
        """Run SMAC on Problem.

        If SMAC is not instantiated, instantiate.
        """
        if self._optimizer is None:
            self._optimizer = self.setup_optimizer()
        self.start_time = time.time()
        while True:
            if self.max_num_trials is not None:
                if self.trial_counter >= self.max_num_trials:
                    break
            if self.wallclock_times is not None:
                if time.time() - self.start_time > self.wallclock_times:
                    break

            syne_trial = self.ask()
            cost = self.evaluate(syne_trial)
            self.tell(syne_trial, cost)

        return None
