"""Nevergrad optimizer.

* Nevergrad Documentation: https://facebookresearch.github.io/nevergrad/
* List of Optimizers available:
* https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers
* Other Optimizers:
* Hyperopt: https://github.com/hyperopt/hyperopt,
* CMA-ES: https://github.com/CMA-ES/pycma,
* bayes_opt: https://github.com/bayesian-optimization/BayesianOptimization
* DE: https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ConfigSpace.hyperparameters as CSH
import nevergrad as ng
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from nevergrad.optimization.base import (
        ConfiguredOptimizer as ConfNGOptimizer,
        Optimizer as NGOptimizer,
    )
    from nevergrad.parametrization import parameter
    from omegaconf import DictConfig

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


opt_list = sorted(ng.optimizers.registry.keys())
ext_opts = {
    "Hyperopt": ng.optimization.optimizerlib.NGOpt13,
    "CMA-ES": ng.optimization.optimizerlib.ParametrizedCMA,
    "bayes_opt": ng.optimization.optimizerlib.ParametrizedBO,
    "DE": ng.families.DifferentialEvolution,
    "EvolutionStrategy": ng.families.EvolutionStrategy,
}


def CS_to_nevergrad_space(hp: CSH.Hyperparameter) -> ng.p.Instrumentation:
    """Convert ConfigSpace to Nevergrad Parameter."""
    if isinstance(hp, CSH.FloatHyperparameter):
        if hp.log:
            return ng.p.Log(lower=hp.lower, upper=hp.upper)
        else:
            return ng.p.Scalar(lower=hp.lower, upper=hp.upper)
    elif isinstance(hp, CSH.IntegerHyperparameter):
        if hp.log:
            return ng.p.Log(lower=hp.lower, upper=hp.upper).set_integer_casting()
        else:
            return ng.p.Scalar(lower=hp.lower, upper=hp.upper).set_integer_casting()
    elif isinstance(hp, CSH.CategoricalHyperparameter):
        return ng.p.Choice(hp.choices)
    elif isinstance(hp, CSH.OrdinalHyperparameter):
        return ng.p.TransitionChoice(hp.sequence)
    elif isinstance(hp, CSH.Constant):
        return ng.p.Choice([hp.value])
    else:
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


class NevergradOptimizer(Optimizer):
    """An optimizer that uses Nevergrad to optimize an objective function."""

    def __init__(
        self,
        problem: Problem,
        nevergrad_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)

        self.fidelity_enabled = False
        self.fidelity_type = None
        if self.task.is_multifidelity:
            self.fidelity_enabled = True
            self.fidelity_type: str = self.task.fidelity_type
        self.task = task
        self.configspace = problem.configspace
        self.ng_space = self.convert_configspace(self.configspace)
        self.nevergrad_cfg = nevergrad_cfg
        self.optimizer_cfg = optimizer_cfg
        if self.optimizer_cfg is None:
            self.optimizer_cfg = {}
        if self.nevergrad_cfg.optimizer_name not in opt_list and self.nevergrad_cfg.optimizer_name not in ext_opts:
            raise ValueError(f"Optimizer {self.nevergrad_cfg.optimizer_name} not found in Nevergrad!")

        self._solver: NGOptimizer | ConfNGOptimizer | None = None
        self.counter = 0
        self.history: dict[str, tuple[ng.p.Parameter, float | list[float] | None]] = {}

    def convert_configspace(self, configspace: ConfigurationSpace) -> ng.p.Parameter:
        """Convert ConfigSpace configuration space to search space from optimizer.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns:
        -------
        SearchSpace
            Optimizer's search space.
        """
        ng_param = ng.p.Dict()
        for hp in configspace.get_hyperparameters():
            ng_param[hp.name] = CS_to_nevergrad_space(hp)
        return ng_param

    def _setup_optimizer(self) -> NGOptimizer | ConfNGOptimizer:
        if self.nevergrad_cfg.optimizer_name in ext_opts:
            if self.nevergrad_cfg.optimizer_name == "Hyperopt":
                ng_opt = ext_opts[self.nevergrad_cfg.optimizer_name]
            else:
                ng_opt = ext_opts[self.nevergrad_cfg.optimizer_name](**self.optimizer_cfg)
            ng_opt = ng_opt(
                parametrization=self.ng_space,
                budget=self.task.n_trials,
                num_workers=self.task.n_workers,
            )
        else:
            ng_opt = ng.optimizers.registry[self.nevergrad_cfg.optimizer_name](
                parametrization=self.ng_space,
                budget=self.task.n_trials,
                num_workers=self.task.n_workers,
            )
        ng_opt.parametrization.random_state = np.random.RandomState(self.nevergrad_cfg.seed)
        return ng_opt

    def convert_to_trial(  # type: ignore[override]
        self,
        config: Configuration,
        name: str | None = None,
        seed: int | None = None,
        budget: float | None = None,
    ) -> TrialInfo:
        """Convert proposal from Nevergrad to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Parameters
        ----------
        config : Configuration
            Configuration from Nevergrad.
        name : str, optional
            Name of the trial, by default None
        seed : int, optional
            Seed of the trial, by default None

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        return TrialInfo(
            config=config,
            name=name,
            seed=self.nevergrad_cfg.seed,
            budget=budget,
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
        config: parameter.Parameter = self.solver.ask()
        unique_name = f"{self.counter}_{config.value}_{self.nevergrad_cfg.seed}"
        self.history[unique_name] = (config, None)
        trial_info = self.convert_to_trial(
            config=Configuration(self.configspace, values=config.value, allow_inactive_with_values=True),
            name=unique_name,
            seed=self.nevergrad_cfg.seed,
            budget=None if not self.fidelity_enabled else self.task.max_budget,
        )
        self.counter += 1
        return trial_info

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
        unique_name = trial_info.name
        assert unique_name is not None
        assert unique_name in self.history
        assert self._solver is not None

        self.history[unique_name] = (self.history[unique_name][0], trial_value.cost)

        self.solver.tell(
            self.history[unique_name][0],
            trial_value.cost,
        )

    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        incumbent = None
        cost = None
        unique_name = None
        if self.task.n_objectives > 1:
            configs = self.solver.pareto_front()
            costs = [param.losses.tolist() for param in configs]
            trial_info = [
                self.convert_to_trial(config=Configuration(self.configspace, values=config.value)) for config in configs
            ]
            trial_value = [TrialValue(cost=cost) for cost in costs]
            incumbent_tuple = list(zip(trial_info, trial_value, strict=False))
        else:
            for name, value in self.history.items():
                if incumbent is None or value[1] < cost:
                    incumbent = value[0].value
                    cost = value[1]
                    unique_name = name
            if cost is None:
                raise ValueError(f"Tried to get Incumbent without calling tell() for config {incumbent}!")
            trial_info = self.convert_to_trial(
                config=Configuration(self.configspace, values=incumbent, allow_inactive_with_values=True),
                name=unique_name,
                seed=self.nevergrad_cfg.seed,
            )
            trial_value = TrialValue(cost=cost)
            incumbent_tuple = (trial_info, trial_value)
        return incumbent_tuple
