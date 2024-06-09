"""Scikit-Optimize.

* Scikit-Optimize optimizer for CARPS.
* NOTE: Set to always minimize the cost
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import skopt
from skopt.space.space import Categorical, Integer, Real, Space

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


base_estimators = ["GP", "RF", "ET", "GBRT"]
acq_funcs = ["LCB", "EI", "PI", "gp_hedge"]
acq_optimizers = ["sampling", "lbfgs", "auto"]


def CS_to_skopt_space(hp: CSH.Hyperparameter) -> Space:
    """Convert ConfigSpace hyperparameter to skopt Space."""
    if isinstance(hp, CSH.FloatHyperparameter):
        if hp.log:
            return Real(hp.lower, hp.upper, name=hp.name, prior="log-uniform")
        else:
            return Real(hp.lower, hp.upper, name=hp.name)
    elif isinstance(hp, CSH.IntegerHyperparameter):
        if hp.log:
            return Integer(hp.lower, hp.upper, name=hp.name, prior="log-uniform")
        else:
            return Integer(hp.lower, hp.upper, name=hp.name)
    elif isinstance(hp, CSH.CategoricalHyperparameter):
        weights = None
        if hp.weights is not None:
            weights = np.asarray(hp.weights) / np.sum(hp.weights)
        return Categorical(hp.choices, name=hp.name, prior=weights)
    elif isinstance(hp, CSH.OrdinalHyperparameter):
        return Categorical(list(hp.sequence), name=hp.name)
    elif isinstance(hp, CSH.Constant):
        return Categorical([hp.value], name=hp.name)
    else:
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


class SkoptOptimizer(Optimizer):
    """An optimizer that uses Scikit-Optimize to optimize an objective function."""

    def __init__(
        self,
        problem: Problem,
        skopt_cfg: DictConfig,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        """Initialize a Scikit-Optimize optimizer."""
        super().__init__(problem, task, loggers)

        self.fidelity_enabled = False

        self.configspace = problem.configspace

        self.skopt_space = self.convert_configspace(self.configspace)
        self._solver: skopt.optimizer.Optimizer | None = None

        self.skopt_cfg = skopt_cfg
        assert self.skopt_cfg.base_estimator in base_estimators
        assert self.skopt_cfg.acq_func in acq_funcs
        assert self.skopt_cfg.acq_optimizer in acq_optimizers

    def convert_configspace(self, configspace: CS.ConfigurationSpace) -> list[Space]:
        """Convert ConfigSpace configuration space to Scikit-Optimize Space.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns:
        -------
        list[Space]
            Scikit-Optimize Space.
        """
        space = []
        for hp in configspace.get_hyperparameters():
            space.append(CS_to_skopt_space(hp))
        return space

    def convert_to_trial(  # type: ignore[override]
        self, config: list[Any]
    ) -> TrialInfo:
        """Convert proposal by Scikit-Optimize to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        configuration = CS.Configuration(
            configuration_space=self.configspace,
            values={hp.name: value for hp, value in zip(self.configspace.get_hyperparameters(), config, strict=False)},
            allow_inactive_with_values=True,
        )
        assert list(configuration.keys()) == list(self.configspace.get_hyperparameter_names())
        assert list(configuration.keys()) == [hp.name for hp in self.skopt_space]
        return TrialInfo(
            config=configuration,
            seed=self.skopt_cfg.random_state,
            budget=None,
            instance=None,
        )

    def _setup_optimizer(self) -> skopt.optimizer.Optimizer:
        if self.skopt_cfg is None:
            self.skopt_cfg = {}
        return skopt.optimizer.Optimizer(dimensions=self.skopt_space, **self.skopt_cfg)

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
        config = self.solver.ask()
        return self.convert_to_trial(config)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
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
        _ = self.solver.tell(list(trial_info.config.values()), trial_value.cost)

    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        best_result = self.solver.get_result()
        best_config = self.convert_to_trial(best_result.x)
        best_cost = TrialValue(cost=best_result.fun)
        return (best_config, best_cost)
