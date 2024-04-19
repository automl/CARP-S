from __future__ import annotations

import copy
import datetime
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter, Constant,
                                         FloatHyperparameter, Hyperparameter,
                                         IntegerHyperparameter,
                                         OrdinalHyperparameter)
from syne_tune.backend.trial_status import Status
from syne_tune.backend.trial_status import Trial as SyneTrial
from syne_tune.backend.trial_status import TrialResult
from syne_tune.config_space import (choice, lograndint, loguniform, ordinal,
                                    randint, uniform)
from syne_tune.optimizer.baselines import (ASHA, BOHB, BORE, DEHB, KDE,
                                           MOBSTER, BayesianOptimization)
from syne_tune.optimizer.scheduler import TrialScheduler as SyneTrialScheduler

from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent


# This is a subset from the syne-tune baselines
optimizers_dict = {
    "BayesianOptimization": BayesianOptimization,
    "ASHA": ASHA,
    "MOBSTER": MOBSTER,
    "BOHB": BOHB,
    "KDE": KDE,
    "BORE": BORE,
    "DEHB": DEHB,
}

mf_optimizer_dicts = {"with_mf": {"ASHA", "DEHB", "MOBSTER", "BOHB"}, "without_mf": {"BORE", "BayesianOptimization", "KDE"}}


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
        n_trials: int | None, 
        time_budget: float | None = None,
        n_workers: int = 1,
        max_budget: float | None = None,
        optimizer_kwargs: dict | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, n_trials, time_budget, n_workers, loggers)
        self.fidelity_enabled = False
        self.max_budget = max_budget

        self.configspace = self.problem.configspace
        assert optimizer_name in optimizers_dict
        if optimizer_name in mf_optimizer_dicts["with_mf"]:
            # raise NotImplementedError("Multi-Fidelity Optimization on SyneTune is not implemented yet!")
            self.fidelity_enabled = True
            if not hasattr(problem, "budget_type"):
                raise ValueError("To run multi-fidelity optimizer, the problem must have a budget_type!")
            if max_budget is None:
                raise ValueError("To run multi-fidelity optimizer, we must specify max_budget!")

        self.syne_tune_configspace = self.convert_configspace(self.configspace)
        self.metric = getattr(problem, "metric", "cost")
        self.budget_type = getattr(self.problem, "budget_type", None)
        self.trial_counter = 0

        self.optimizer_name = optimizer_name
        self._solver: SyneTrialScheduler | None = None 

        self.optimizer_kwargs = optimizer_kwargs

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
        if self.fidelity_enabled:
            configspace_st[self.problem.budget_type] = self.max_budget
        return configspace_st

    def convert_to_trial(  # type: ignore[override]
        self, trial: SyneTrial
    ) -> TrialInfo:
        """Convert proposal from SyneTune to TrialInfo.

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
        configs = copy.deepcopy(trial.config)
        if self.budget_type is not None:
            budget = configs.pop(self.budget_type)
        else:
            budget = None
        configuration = Configuration(configuration_space=self.configspace, values=configs)
        trial_info = TrialInfo(config=configuration, seed=None, budget=budget, instance=None)
        return trial_info

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Returns
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        trial_suggestion = self._solver.suggest(self.trial_counter)
        trial = SyneTrial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        trial_info = self.convert_to_trial(trial=trial)
        return trial_info
    
    def convert_to_synetrial(self, trial_info: TrialInfo) -> SyneTrial:
        """Convert a trial info to the syne tune format.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info.

        Returns
        -------
        SyneTrial
            Synetune format
        """
        syne_config = dict(trial_info.config)
        if self.fidelity_enabled:
            syne_config[self.problem.budget_type] = trial_info.budget
        trial = SyneTrial(
            trial_id=self.trial_counter,
            config=syne_config,
            creation_time=datetime.datetime.now(),
        )
        return trial

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue):
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        cost = trial_value.cost
        trial = self.convert_to_synetrial(trial_info=trial_info)
        experiment_result = {self.metric: cost}
        self.trial_counter += 1
        self._solver.on_trial_complete(trial=trial, result=experiment_result)
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.datetime.now(),
        )
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metric: str) -> TrialResult:
        """
        Return the best trial according to the provided metric
        """
        if self.solver.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metric],
        )

    def _setup_optimizer(self) -> SyneTrialScheduler:
        """
        Setup Optimizer.

        Retrieve defaults and instantiate SyneTune.

        Returns
        -------
        SyneTrialScheduler
            Instance of a SyneTune.

        """
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        _optimizer_kwargs = dict(
            config_space=self.syne_tune_configspace,
            metric=getattr(self.problem, "metric", "cost"),
            mode="min",
        )
        if self.optimizer_name in mf_optimizer_dicts["with_mf"]:
            _optimizer_kwargs["resource_attr"] = self.problem.budget_type
            # _optimizer_kwargs["max_t"] = self.max_budget  # TODO check how to set n trials / wallclock limit for synetune

        self.optimizer_kwargs.update(_optimizer_kwargs)

        bscheduler = optimizers_dict[self.optimizer_name](**self.optimizer_kwargs)
        return bscheduler
    
    def get_current_incumbent(self) \
            -> Incumbent:
        trial_result = self.best_trial(metric=self.metric)
        trial_info = self.convert_to_trial(trial=trial_result)
        cost = trial_result.metrics[self.metric]
        trial_value = TrialValue(
            cost=cost,
            time=trial_result.seconds,
            virtual_time=trial_result.seconds,
            additional_info=trial_result.metrics
        )
        return (trial_info, trial_value)
