"""DEHB Optimizer.

* Source: https://github.com/automl/DEHB/tree/master
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dehb import DEHB

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from omegaconf import DictConfig

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


class DEHBOptimizer(Optimizer):
    """An optimizer that uses DEHB to optimize an objective function."""

    def __init__(
        self,
        problem: Problem,
        dehb_cfg: DictConfig,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(problem, task, loggers)

        self.fidelity_enabled = True
        self.task = task
        self.dehb_cfg = dehb_cfg
        self.configspace = self.convert_configspace(problem.configspace)
        self.configspace.seed(dehb_cfg.seed)
        if self.task.max_budget is None:
            raise ValueError("max_budget must be specified to run DEHB!")
        if self.task.min_budget is None:
            raise ValueError("min_budget must be specified to run DEHB!")
        self._solver: DEHB | None = None
        self.history: dict[str, dict[str, Any]] = {}

    def _setup_optimizer(self) -> Any:
        return DEHB(
            cs=self.configspace,
            min_fidelity=self.task.min_budget,
            max_fidelity=self.task.max_budget,
            n_workers=self.task.n_workers,
            **self.dehb_cfg,
        )

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
        self,
        config: Configuration,
        name: str | None = None,
        seed: int | None = None,
        budget: float | None = None,
    ) -> TrialInfo:
        """Convert proposal from DEHB to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Parameters
        ----------
        config : Configuration
            Configuration from DEHB.
        name : str, optional
            Name of the trial, by default None
        seed : int, optional
            Seed of the trial, by default None
        budget : float, optional
            Budget of the trial, by default None

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        return TrialInfo(config=config, name=name, seed=seed, budget=budget)

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
        info = self.solver.ask()
        unique_name = f"{info['config_id']}_{info['fidelity']}_{self.dehb_cfg.seed}"
        self.history[unique_name] = info
        return self.convert_to_trial(
            config=info["config"],
            name=unique_name,
            seed=self.dehb_cfg.seed,
            budget=info["fidelity"],
        )

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

        dehb_job_info = self.history[unique_name]
        if isinstance(trial_value.cost, list):
            raise NotImplementedError("Multiobjective optimization not yet implemented for DEHB!")
        dehb_result = {"fitness": float(trial_value.cost), "cost": (trial_value.time)}
        self.solver.tell(dehb_job_info, dehb_result)

    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        incumbent = self.solver.get_incumbents()
        inc_config = self.convert_to_trial(
            config=incumbent[0],
            seed=self.dehb_cfg.seed,
        )
        inc_value = TrialValue(cost=incumbent[1])
        return (inc_config, inc_value)
