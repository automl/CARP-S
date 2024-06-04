from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from carps.optimizers.optimizer import Optimizer
from carps.utils.exceptions import AskAndTellNotSupportedError
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from smac.facade.smac_ac_facade import SMAC4AC

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


class NotSupportedError(Exception):
    pass


class SMAC314Optimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        smac_cfg: DictConfig,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)

        self.configspace = self.problem.configspace
        self.smac_cfg = smac_cfg
        self._solver: SMAC4AC | None = None

    def convert_configspace(self, configspace: ConfigurationSpace) -> ConfigurationSpace:
        """Convert configuration space from Problem to Optimizer.

        Here, we don't need to convert.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns:
        -------
        ConfigurationSpace
            Configuration space for Optimizer.
        """
        return configspace

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

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        return TrialInfo(config=config, seed=seed, budget=budget, instance=instance)

    def target_function(
        self, config: Configuration, seed: int | None = None, budget: float | None = None, instance: str | None = None
    ) -> float | list[float]:
        """Target Function.

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

        Returns:
        -------
        float | list[float]
            Cost as float or list[float], depending on the number of objectives.
        """
        trial_info = self.convert_to_trial(config=config, seed=seed, budget=budget, instance=instance)
        trial_value = self.problem.evaluate(trial_info=trial_info)
        return trial_value.cost

    def _setup_optimizer(self) -> SMAC4AC:
        """Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns:
        -------
        SMAC4AC
            Instance of a SMAC facade.

        """
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.facade.smac_bb_facade import SMAC4BB
        from smac.facade.smac_hpo_facade import SMAC4HPO
        from smac.facade.smac_mf_facade import SMAC4MF
        from smac.scenario.scenario import Scenario

        if self.smac_cfg.scenario.n_workers > 1 and self.smac_cfg.optimization_type != "mf":
            raise NotSupportedError("SMAC 1.4 does not support parallel execution natively.")

        if self.smac_cfg.scenario.wallclock_limit is None:
            self.smac_cfg.scenario.wallclock_limit = np.inf

        # Instantiate Scenario
        scenario_kwargs = {
            "cs": self.configspace,
            "output_dir": None,
        }
        # We always expect scenario kwargs from the user
        _scenario_kwargs = OmegaConf.to_container(self.smac_cfg.scenario, resolve=True)
        scenario_kwargs.update(_scenario_kwargs)

        scenario = Scenario(scenario=scenario_kwargs)

        intensifier_kwargs: dict[Any, Any] = {}
        facade_kwargs: dict[Any, Any] = {}

        # Create facade
        if self.smac_cfg.optimization_type == "bb":
            facade_object = SMAC4BB
            intensifier_kwargs["maxR"] = self.smac_cfg.max_config_calls

        elif self.smac_cfg.optimization_type == "hpo":
            facade_object = SMAC4HPO
            intensifier_kwargs["maxR"] = self.smac_cfg.max_config_calls

        elif self.smac_cfg.optimization_type == "mf":
            facade_object = SMAC4MF

            n_seeds = self.smac_cfg.get("n_seeds", None)
            intensifier_kwargs["n_seeds"] = n_seeds
            intensifier_kwargs["initial_budget"] = self.smac_cfg.scenario.min_budget
            intensifier_kwargs["max_budget"] = self.smac_cfg.scenario.max_budget

            inc_selection = self.smac_cfg.incumbent_selection
            if inc_selection == "highest_observed_budget":
                inc_selection = "highest_budget"

            intensifier_kwargs["incumbent_selection"] = inc_selection
            facade_kwargs["n_jobs"] = self.smac_cfg.scenario.n_workers
        elif self.smac_cfg.optimization_type == "ac":
            facade_object = SMAC4AC
            intensifier_kwargs["maxR"] = self.smac_cfg.max_config_calls

        else:
            raise RuntimeError("Unknown optimization type.")

        if self.smac_cfg.intensifier is None:
            intensifier = None
        elif self.smac_cfg.intensifier == "successive_halving":
            from smac.intensification.successive_halving import SuccessiveHalving

            intensifier = SuccessiveHalving
        else:
            raise RuntimeError("Unsupported intensifier.")

        return facade_object(
            scenario=scenario,
            tae_runner=self.target_function,
            intensifier=intensifier,
            intensifier_kwargs=intensifier_kwargs,
            **facade_kwargs,
        )

    def ask(self) -> TrialInfo:
        raise AskAndTellNotSupportedError

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        raise AskAndTellNotSupportedError

    def _run(self) -> Incumbent:
        """Run SMAC on Problem."""
        incumbent = self.solver.optimize()  # noqa: F841
        return self.get_current_incumbent()

    def get_current_incumbent(self) -> Incumbent:
        trial_info = TrialInfo(config=self.solver.solver.incumbent)
        trial_value = TrialValue(cost=self.solver.get_runhistory().get_cost(self.solver.solver.incumbent))
        return (trial_info, trial_value)
