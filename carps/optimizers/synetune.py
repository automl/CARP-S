from __future__ import annotations

import copy
import datetime
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import omegaconf
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from syne_tune.backend.trial_status import (
    Status,
    Trial as SyneTrial,
    TrialResult,
)
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
    SyncMOBSTER,
    MOREA,
    BayesianOptimization,
    MOLinearScalarizationBayesOpt,
    MORandomScalarizationBayesOpt,
)

from carps.optimizers.optimizer import Optimizer
from carps.utils.pareto_front import pareto
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from syne_tune.optimizer.scheduler import TrialScheduler as SyneTrialScheduler

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
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
    "SyncMOBSTER": SyncMOBSTER,
}

mf_optimizer_dicts = {
    "with_mf": {"ASHA", "MOASHA", "DEHB", "MOBSTER", "BOHB", "SyncMOBSTER"},
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
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


class SynetuneOptimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        optimizer_name: str,
        task: Task,
        optimizer_kwargs: dict | None = None,
        loggers: list[AbstractLogger] | None = None,
        conversion_factor: int = 1000,
    ) -> None:
        super().__init__(problem, task, loggers)
        self.fidelity_enabled = False
        self.max_budget = task.max_budget
        assert optimizer_name in optimizers_dict
        if optimizer_name in mf_optimizer_dicts["with_mf"]:
            # raise NotImplementedError("Multi-Fidelity Optimization on SyneTune is not implemented yet!")
            self.fidelity_enabled = True
            if self.task.fidelity_type is None:
                raise ValueError("To run multi-fidelity optimizer, the problem must define a fidelity type!")
            if self.max_budget is None:
                raise ValueError("To run multi-fidelity optimizer, we must specify max_budget!")

        self.fidelity_type: str = self.task.fidelity_type
        self.configspace = self.problem.configspace
        self.metric: str | list[str] = self.task.objectives
        if len(self.metric) == 1:
            self.metric = self.metric[0]
        self.conversion_factor = conversion_factor
        self.trial_counter: int = 0

        self.optimizer_name = optimizer_name
        self._solver: SyneTrialScheduler | None = None

        self.optimizer_kwargs = (
            omegaconf.OmegaConf.to_object(optimizer_kwargs) if optimizer_kwargs is not None else None
        )

        self.completed_experiments: OrderedDict[int, TrialResult] = OrderedDict()
        self.convert = False

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
            configspace_st[self.fidelity_type] = (
                self.max_budget if not self.convert else int(self.conversion_factor * self.max_budget)
            )

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

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        configs = copy.deepcopy(trial.config)
        budget = configs.pop(self.fidelity_type) if self.fidelity_type is not None else None
        configuration = Configuration(
            configuration_space=self.configspace, values=configs, allow_inactive_with_values=True
        )
        return TrialInfo(
            config=configuration,
            seed=None,
            budget=budget if not self.convert else budget / self.conversion_factor,
            instance=None,
        )

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Returns:
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
        return self.convert_to_trial(trial=trial)

    def convert_to_synetrial(self, trial_info: TrialInfo) -> SyneTrial:
        """Convert a trial info to the syne tune format.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info.

        Returns:
        -------
        SyneTrial
            Synetune format
        """
        syne_config = dict(trial_info.config)
        if self.fidelity_enabled:
            syne_config[self.fidelity_type] = (
                trial_info.budget if not self.convert else int(self.conversion_factor * trial_info.budget)
            )
        return SyneTrial(
            trial_id=self.trial_counter,
            config=syne_config,
            creation_time=datetime.datetime.now(),
        )

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
        experiment_result = {}
        if self.task.n_objectives == 1:
            experiment_result = {self.task.objectives[0]: cost}
        else:
            experiment_result = {self.task.objectives[i]: cost[i] for i in range(len(cost))}

        if self.task.is_multifidelity:
            experiment_result[self.fidelity_type] = (
                trial_info.budget if not self.convert else int(self.conversion_factor * trial_info.budget)
            )
            # del experiment_result[self.task.objectives]

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
        """Return the best trial according to the provided metric."""
        if self.optimizer_name == "MOASHA":
            self.solver.mode = "min"

        sign = 1.0 if self.solver.mode == "max" else -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metric],
        )

    def get_pareto_front(self) -> list[TrialResult]:
        """Return the pareto front for multi-objective optimization."""
        if self.task.is_multifidelity:
            # Determine maximum budget run
            max_budget = np.max([v.metrics[self.fidelity_type] for v in self.completed_experiments.values()])
            # Get only those trial results that ran on max budget
            results_on_highest_fidelity = np.array(
                [v for v in self.completed_experiments.values() if v.metrics[self.fidelity_type] == max_budget]
            )
            # Get costs, exclude fidelity
            costs = np.array([[v.metrics[m] for m in self.task.objectives] for v in results_on_highest_fidelity])
            # Determine pareto front of the trials run on max budget
            front = results_on_highest_fidelity[pareto(costs)]
        else:
            results = np.array(list(self.completed_experiments.values()))
            costs = np.array([list(trial.metrics.values()) for trial in results])
            front = results[pareto(costs)]
        return front.tolist()

    def _setup_optimizer(self) -> SyneTrialScheduler:
        """Setup Optimizer.

        Retrieve defaults and instantiate SyneTune.

        Returns:
        -------
        SyneTrialScheduler
            Instance of a SyneTune.

        """
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}

        _optimizer_kwargs = {
            "metric": self.metric,
            "mode": "min" if self.task.n_objectives == 1 else list(np.repeat("min", self.task.n_objectives)),
        }

        if self.optimizer_name in mf_optimizer_dicts["with_mf"]:
            _optimizer_kwargs["resource_attr"] = self.fidelity_type
            # _optimizer_kwargs["max_t"] = self.max_budget  # TODO check how to set n trials / wallclock limit for synetune

            # for floating point resources like trainsize, we need to convert them to integer for synetune
            if "grace_period" in self.optimizer_kwargs and isinstance(self.optimizer_kwargs["grace_period"], float):
                self.convert = True
                _optimizer_kwargs["grace_period"] = int(self.conversion_factor * self.optimizer_kwargs["grace_period"])

                if "max_t" in self.optimizer_kwargs:
                    _optimizer_kwargs["max_t"] = int(self.conversion_factor * self.max_budget)

                if "max_resource_level" in self.optimizer_kwargs:
                    _optimizer_kwargs["max_resource_level"] = int(
                        self.conversion_factor * self.optimizer_kwargs["max_resource_level"]
                    )

            print(_optimizer_kwargs)

        self.syne_tune_configspace = self.convert_configspace(self.configspace)
        _optimizer_kwargs["config_space"] = self.syne_tune_configspace

        self.optimizer_kwargs.update(_optimizer_kwargs)

        if self.optimizer_name == "MOASHA":
            self.metric = self.optimizer_kwargs["metric"]
            del self.optimizer_kwargs["metric"]
            del self.optimizer_kwargs["resource_attr"]

        if self.optimizer_name in ["SyncMOBSTER"]:
            # del self.optimizer_kwargs["metric"]
            if "time_attr" in self.optimizer_kwargs:
                del self.optimizer_kwargs["time_attr"]

        return optimizers_dict[self.optimizer_name](**self.optimizer_kwargs)

    def get_current_incumbent(self) -> Incumbent:
        if self.task.n_objectives == 1:
            trial_result = self.best_trial(metric=self.task.objectives[0])
            trial_info = self.convert_to_trial(trial=trial_result)
            cost = trial_result.metrics[self.task.objectives[0]]
            trial_value = TrialValue(
                cost=cost,
                time=trial_result.seconds,
                virtual_time=trial_result.seconds,
                additional_info=trial_result.metrics,
            )
            incumbent_tuple = (trial_info, trial_value)
        else:
            trial_result = self.get_pareto_front()
            tis, tvs = [], []
            for result in trial_result:
                trial_info = self.convert_to_trial(trial=result)
                costs = list(result.metrics.values())
                trial_value = TrialValue(
                    cost=costs, time=result.seconds, virtual_time=result.seconds, additional_info=result.metrics
                )
                tis.append(trial_info)
                tvs.append(trial_value)
            incumbent_tuple = list(zip(tis, tvs, strict=False))
        return incumbent_tuple
