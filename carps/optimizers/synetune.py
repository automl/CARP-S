from __future__ import annotations

import copy
import datetime
from collections import OrderedDict
from typing import Any, Callable

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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from syne_tune.backend.trial_status import Status
from syne_tune.backend.trial_status import Trial as SyneTrial
from syne_tune.backend.trial_status import TrialResult
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
    MOASHA,
    MOBSTER,
    MOREA,
    BayesianOptimization,
    MOLinearScalarizationBayesOpt,
    MORandomScalarizationBayesOpt,
)
from syne_tune.optimizer.scheduler import TrialScheduler as SyneTrialScheduler

from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer
from carps.utils.task import Task
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent

# This is a subset from the syne-tune baselines
optimizers_dict = {
    "BayesianOptimization": BayesianOptimization,
    "BO-MO-RS": MORandomScalarizationBayesOpt,
    "BO-MO-LS": MOLinearScalarizationBayesOpt,
    "MOREA": MOREA,
    "ASHA": ASHA,
    "MOBSTER": MOBSTER,
    "BOHB": BOHB,
    "KDE": KDE,
    "BORE": BORE,
    "DEHB": DEHB,
    "MOASHA": MOASHA,
}

mf_optimizer_dicts = {
    "with_mf": {"ASHA", "MOASHA", "DEHB", "MOBSTER", "BOHB"},
    "without_mf": {"BORE", "BayesianOptimization", "KDE"},
}


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
        raise NotImplementedError(
            f"Unknown hyperparameter type: {hp.__class__.__name__}"
        )


class SynetuneOptimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        optimizer_name: "str",
        task: Task,
        optimizer_kwargs: dict | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)
        self.fidelity_enabled = False
        self.max_budget = task.max_budget
        assert optimizer_name in optimizers_dict
        if optimizer_name in mf_optimizer_dicts["with_mf"]:
            # raise NotImplementedError("Multi-Fidelity Optimization on SyneTune is not implemented yet!")
            self.fidelity_enabled = True
            if self.task.fidelity_type is None:
                raise ValueError(
                    "To run multi-fidelity optimizer, the problem must define a fidelity type!"
                )
            if self.max_budget is None:
                raise ValueError(
                    "To run multi-fidelity optimizer, we must specify max_budget!"
                )

        self.fidelity_type: str = self.task.fidelity_type
        self.configspace = self.problem.configspace
        self.syne_tune_configspace = self.convert_configspace(self.configspace)
        self.metric = getattr(problem, "metric", "cost")
        self.trial_counter: int = 0

        self.optimizer_name = optimizer_name
        self._solver: SyneTrialScheduler | None = None

        self.optimizer_kwargs = optimizer_kwargs

        self.completed_experiments: OrderedDict[
            int, tuple[TrialValue, TrialInfo]
        ] = OrderedDict()

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
            configspace_st[self.fidelity_type] = self.max_budget
        return configspace_st

    def convert_to_trial(self, trial: SyneTrial) -> TrialInfo:  # type: ignore[override]
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
        if self.fidelity_type is not None:
            budget = configs.pop(self.fidelity_type)
        else:
            budget = None
        configuration = Configuration(
            configuration_space=self.configspace, values=configs
        )
        trial_info = TrialInfo(
            config=configuration, seed=None, budget=budget, instance=None
        )
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
            syne_config[self.fidelity_type] = trial_info.budget
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
        if self.optimizer_name == "MOASHA":
            for m, c in zip(self.metric, cost):
                experiment_result[m] = c
            experiment_result[self.fidelity_type] = trial_info.budget
            del experiment_result[self.metric]

            self._solver.on_trial_add(trial=trial)

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
        if self.optimizer_name == "MOASHA":
            self.solver.mode = "min"

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
            _optimizer_kwargs["resource_attr"] = self.fidelity_type
            # _optimizer_kwargs["max_t"] = self.max_budget  # TODO check how to set n trials / wallclock limit for synetune

        self.optimizer_kwargs.update(_optimizer_kwargs)

        if self.optimizer_name == "MOASHA":
            self.metric = self.optimizer_kwargs["metrics"]
            del self.optimizer_kwargs["metric"]
            del self.optimizer_kwargs["resource_attr"]

        bscheduler = optimizers_dict[self.optimizer_name](**self.optimizer_kwargs)
        return bscheduler

    def get_current_incumbent(self) -> Incumbent:
        if isinstance(self.metric, str):
            trial_result = self.best_trial(metric=self.metric)
            trial_info = self.convert_to_trial(trial=trial_result)
            cost = trial_result.metrics[self.metric]
            trial_value = TrialValue(
                cost=cost,
                time=trial_result.seconds,
                virtual_time=trial_result.seconds,
                additional_info=trial_result.metrics,
            )
            return (trial_info, trial_value)

        else:
            # multiobjecti
            max_budget = np.max(
                [
                    v.metrics[self.fidelity_type]
                    for v in self.completed_experiments.values()
                ]
            )
            highest_fidelity = [
                v
                for v in self.completed_experiments.values()
                if v.metrics[self.fidelity_type] == max_budget
            ]
            hf_cost = np.array(
                [[v.metrics[m] for m in self.metric] for v in highest_fidelity]
            )

            # # calculate the hypervolume on the highest fidelity!
            # if not hasattr(self, 'ref_point'):
            #     # calculate the reference point as relative margin of the highest fidelity points
            #     ref_point = hf_cost.max(axis=0)
            #     self.hv = self.HV(ref_point=ref_point)
            # hv = self.hv(hf_cost)
            non_dom = NonDominatedSorting().do(hf_cost, only_non_dominated_front=True)

            trial_results = [
                self.completed_experiments[key]
                for index, key in enumerate(self.completed_experiments.keys())
                if index in non_dom
            ]

            incumbents = [
                (
                    self.convert_to_trial(trial=trial_result),
                    TrialValue(
                        cost=[trial_result.metrics[m] for m in self.metric],
                        time=trial_result.seconds,
                        virtual_time=trial_result.seconds,
                        additional_info=None,
                    ),
                )
                for trial_result in trial_results
            ]

            return incumbents
