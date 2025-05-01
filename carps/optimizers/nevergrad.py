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

import ConfigSpace.hyperparameters as CSH  # noqa: N812
import nevergrad as ng
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig, OmegaConf

from carps.optimizers.optimizer import Optimizer
from carps.utils.configuration import clip_bounds
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from nevergrad.optimization.base import (  # type: ignore
        ConfiguredOptimizer as ConfNGOptimizer,
        Optimizer as NGOptimizer,
    )
    from nevergrad.parametrization import parameter  # type: ignore

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


def configspace_hp_to_nevergrad_hp(hp: CSH.Hyperparameter) -> ng.p.Instrumentation:  # noqa: PLR0911
    """Convert ConfigSpace to Nevergrad Parameter.

    Parameters
    ----------
    hp : CSH.Hyperparameter
        Hyperparameter from ConfigSpace.

    Returns:
    -------
    ng.p.Instrumentation
        Hyperparameter from Nevergrad.
    """
    if isinstance(hp, CSH.FloatHyperparameter):
        if hp.log:
            return ng.p.Log(lower=hp.lower, upper=hp.upper)
        return ng.p.Scalar(lower=hp.lower, upper=hp.upper)
    if isinstance(hp, CSH.IntegerHyperparameter):
        if hp.log:
            return ng.p.Log(lower=hp.lower, upper=hp.upper).set_integer_casting()
        return ng.p.Scalar(lower=hp.lower, upper=hp.upper).set_integer_casting()
    if isinstance(hp, CSH.CategoricalHyperparameter):
        return ng.p.Choice(hp.choices)
    if isinstance(hp, CSH.OrdinalHyperparameter):
        return ng.p.TransitionChoice(hp.sequence)
    if isinstance(hp, CSH.Constant):
        return ng.p.Choice([hp.value])
    raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


class NevergradOptimizer(Optimizer):
    """An optimizer that uses Nevergrad to optimize an objective function."""

    def __init__(
        self,
        task: Task,
        nevergrad_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the optimizer.

        Parameters
        ----------
        task : Task
            The task (objective function with specific input and output space and optimization resources) to optimize.
        nevergrad_cfg : DictConfig
            The configuration for the Nevergrad optimizer.
        optimizer_cfg : DictConfig
            The configuration for the optimizer.
        loggers : list[AbstractLogger] | None
        expects_multiple_objectives : bool, optional
            Metadata. Whether the optimizer expects multiple objectives, by default False.
        expects_fidelities : bool, optional
            Metadata. Whether the optimizer expects fidelities for multi-fidelity, by default False.
        """
        super().__init__(
            task,
            loggers,
            expects_fidelities=expects_fidelities,
            expects_multiple_objectives=expects_multiple_objectives,
        )

        self.fidelity_enabled = False
        self.fidelity_type: str | None = None
        if self.task.input_space.fidelity_space.is_multifidelity:
            self.fidelity_enabled = True
            self.fidelity_type = self.task.input_space.fidelity_space.fidelity_type
        self.configspace = task.input_space.configuration_space
        self.ng_space = self.convert_configspace(self.configspace)
        self.nevergrad_cfg = nevergrad_cfg
        self.optimizer_cfg = optimizer_cfg
        if self.optimizer_cfg is None:
            self.optimizer_cfg = {}
        if isinstance(self.optimizer_cfg, DictConfig):
            self.optimizer_cfg = OmegaConf.to_container(self.optimizer_cfg)
        if "scale" in self.optimizer_cfg:
            self.optimizer_cfg["scale"] = float(self.optimizer_cfg["scale"])
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
            Configuration space from ObjectiveFunction.

        Returns:
        -------
        SearchSpace
            Optimizer's search space.
        """
        ng_param = ng.p.Dict()
        for hp in configspace.get_hyperparameters():
            ng_param[hp.name] = configspace_hp_to_nevergrad_hp(hp)
        return ng_param

    def _setup_optimizer(self) -> NGOptimizer | ConfNGOptimizer:
        if self.nevergrad_cfg.optimizer_name in ext_opts:
            if self.nevergrad_cfg.optimizer_name == "Hyperopt":
                ng_opt = ext_opts[self.nevergrad_cfg.optimizer_name]
            else:
                ng_opt = ext_opts[self.nevergrad_cfg.optimizer_name](**self.optimizer_cfg)
            ng_opt = ng_opt(
                parametrization=self.ng_space,
                budget=self.task.optimization_resources.n_trials,
                num_workers=self.task.optimization_resources.n_workers,
            )
        else:
            ng_opt = ng.optimizers.registry[self.nevergrad_cfg.optimizer_name](
                parametrization=self.ng_space,
                budget=self.task.optimization_resources.n_trials,
                num_workers=self.task.optimization_resources.n_workers,
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

        This ensures that the objective function can be evaluated with a unified API.

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
            seed=seed,
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
            config=self.convert_nevergrad_config_to_configspace_config(config),
            name=unique_name,
            seed=self.nevergrad_cfg.seed,
            budget=None if not self.fidelity_enabled else self.task.input_space.fidelity_space.max_fidelity,
        )
        self.counter += 1
        return trial_info

    def convert_nevergrad_config_to_configspace_config(self, config_nevergrad: parameter.Parameter) -> Configuration:
        """Convert nevergrad config to configspace config.

        Might clip to hyperparameter bounds of float hyperparameters to prevent numerical issues after log
        transformation.

        Args:
            config_nevergrad : parameter.Parameter
                The configuration to convert.

        Returns:
            Configuration
                The ConfigSpace configuration.
        """
        config_dict = clip_bounds(config_nevergrad.value, self.configspace)
        return Configuration(self.configspace, values=config_dict, allow_inactive_with_values=True)

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
        if self.task.output_space.n_objectives > 1:
            configs = self.solver.pareto_front()
            costs = [param.losses.tolist() for param in configs]
            trial_infos = [
                self.convert_to_trial(config=self.convert_nevergrad_config_to_configspace_config(config))
                for config in configs
            ]
            trial_values = [TrialValue(cost=cost) for cost in costs]
            incumbent_tuple = list(zip(trial_infos, trial_values, strict=False))
        else:
            incumbent = None
            cost = None
            unique_name = None
            incumbent_config = None
            for name, value in self.history.items():
                if incumbent is None or value[1] < cost:
                    incumbent = value[0].value
                    cost = value[1]
                    unique_name = name
                    incumbent_config = value[0]
            if cost is None:
                raise ValueError(f"Tried to get Incumbent without calling tell() for config {incumbent}!")
            trial_info = self.convert_to_trial(
                config=self.convert_nevergrad_config_to_configspace_config(incumbent_config),
                name=unique_name,
                seed=self.nevergrad_cfg.seed,
            )
            trial_value = TrialValue(cost=cost)
            incumbent_tuple = (trial_info, trial_value)  # type: ignore[assignment]
        return incumbent_tuple
