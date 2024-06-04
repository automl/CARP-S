from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from rich import print as printr

# from git import Repo
# from smac.callback.metadata_callback import MetadataCallback
from smac.multi_objective.parego import ParEGO
from smac.runhistory import (
    TrialInfo as SmacTrialInfo,
    TrialKey as SmacTrialKey,
    TrialValue as SmacTrialValue,
)
from smac.scenario import Scenario

from carps.optimizers.optimizer import Optimizer
from carps.utils.loggingutils import setup_logging
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from smac.facade.abstract_facade import AbstractFacade

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent

setup_logging()


def maybe_inst_add_scenario(smac_kwargs: dict[str, Any], key: str, scenario: Scenario) -> dict[str, Any]:
    if key in smac_kwargs:
        smac_kwargs[key] = smac_kwargs[key](scenario=scenario)
    return smac_kwargs


class SMAC3Optimizer(Optimizer):
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
        self._solver: AbstractFacade | None = None
        self._cb_on_start_called: bool = False

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

    def _setup_optimizer(self) -> AbstractFacade:
        """Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns:
        -------
        SMAC4AC
            Instance of a SMAC facade.

        """
        # repo = Repo(".", search_parent_directories=True)
        # metadata_callback = MetadataCallback(
        #     repository=repo.working_tree_dir.split("/")[-1],
        #     branch=str(repo.active_branch),
        #     commit=str(repo.head.commit),
        #     command=" ".join([sys.argv[0][len(repo.working_tree_dir) + 1 :]] + sys.argv[1:]),
        #     additional_information={
        #         "cfg": OmegaConf.to_yaml(cfg=cfg),
        #     },
        # )

        # Select SMAC Facade
        smac_class = get_class(self.smac_cfg.smac_class)

        if smac_class == get_class("smac.facade.multi_fidelity_facade.MultiFidelityFacade"):
            self.fidelity_enabled = True

        # Setup other SMAC kwargs
        smac_kwargs = {}
        if self.smac_cfg.smac_kwargs is not None:
            smac_kwargs = OmegaConf.to_container(self.smac_cfg.smac_kwargs, resolve=True, enum_to_str=True)

        # Instantiate Scenario
        scenario_kwargs = {
            "configspace": self.configspace,
            # output_directory=Path(self.config.hydra.sweep.dir)
            # / "smac3_output",  # output directory is automatically set via config file
        }
        # We always expect scenario kwargs from the user
        _scenario_kwargs = OmegaConf.to_container(self.smac_cfg.scenario, resolve=True)
        scenario_kwargs.update(_scenario_kwargs)

        scenario = Scenario(**scenario_kwargs)

        smac_kwargs["scenario"] = scenario

        # Convert callbacks to list if necessary
        # Callbacks can come as a dict due to impossible hydra composition of
        # lists.
        if "callbacks" not in smac_kwargs:
            smac_kwargs["callbacks"] = []
        elif "callbacks" in smac_kwargs and type(smac_kwargs["callbacks"]) == dict:
            smac_kwargs["callbacks"] = list(smac_kwargs["callbacks"].values())
        elif "callbacks" in smac_kwargs and type(smac_kwargs["callbacks"]) == list:
            pass

        # If we have a custom intensifier we need to instantiate ourselves
        # because the helper methods in the facades expect a scenario.
        smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "intensifier", scenario)

        if "acquisition_function" in smac_kwargs and "acquisition_maximizer" in smac_kwargs:
            if "acquisition_maximizer" in smac_kwargs:
                smac_kwargs["acquisition_maximizer"] = smac_kwargs["acquisition_maximizer"](
                    configspace=self.configspace, acquisition_function=smac_kwargs["acquisition_function"]
                )
                if hasattr(smac_kwargs["acquisition_maximizer"], "selector") and hasattr(
                    smac_kwargs["acquisition_maximizer"].selector, "expl2callback"
                ):
                    smac_kwargs["callbacks"].append(smac_kwargs["acquisition_maximizer"].selector.expl2callback)

        smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "config_selector", scenario)
        smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "initial_design", scenario)
        smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "multi_objective_algorithm", scenario)

        printr(smac_class, smac_kwargs)

        smac = smac_class(
            target_function=self.target_function,
            **smac_kwargs,
        )
        printr(smac)

        return smac

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
        if not self._cb_on_start_called:
            self._cb_on_start_called = True
            for callback in self.solver.optimizer._callbacks:
                callback.on_start(self.solver.optimizer)
        smac_trial_info = self.solver.ask()
        return TrialInfo(
            config=smac_trial_info.config,
            instance=smac_trial_info.instance,
            budget=smac_trial_info.budget,
            seed=smac_trial_info.seed,
            name=None,
            checkpoint=None,
        )

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        smac_trial_info = SmacTrialInfo(
            config=trial_info.config, instance=trial_info.instance, budget=trial_info.budget, seed=trial_info.seed
        )
        additional_info = trial_value.additional_info
        if self.task.n_objectives > 1:
            # Save costs for multi-objective because SMAC might scalarize the cost
            additional_info["cost"] = trial_value.cost
        smac_trial_value = SmacTrialValue(
            cost=trial_value.cost,
            time=trial_value.time,
            status=trial_value.status,
            starttime=trial_value.starttime,
            endtime=trial_value.endtime,
            additional_info=additional_info,
        )
        self.solver.tell(info=smac_trial_info, value=smac_trial_value)

    def get_current_incumbent(self) -> Incumbent:
        if self.solver.scenario.count_objectives() == 1:
            inc = self.solver.intensifier.get_incumbent()
            cost = self.solver.runhistory.get_cost(config=inc)
            trial_info = TrialInfo(config=inc)
            trial_value = TrialValue(cost=cost)
            incumbent_tuple = (trial_info, trial_value)
        else:
            if (mo := self.solver._runhistory_encoder.multi_objective_algorithm) is not None:
                if isinstance(mo, ParEGO) and mo._theta is None:
                    # Initialize weights of ParEGO
                    mo.update_on_iteration_start()
            incs = self.solver.intensifier.get_incumbents()
            tis = [
                self.solver.runhistory.get_trials(c)[0] for c in incs
            ]  # first trial because only one budget #TODO update for momf
            tks = [
                SmacTrialKey(
                    config_id=self.solver.runhistory.get_config_id(ti.config),
                    instance=ti.instance,
                    seed=ti.seed,
                    budget=ti.budget,
                )
                for ti in tis
            ]
            tvs = [self.solver.runhistory[k] for k in tks]
            tvs = [TrialValue(**asdict(tv)) for tv in tvs]
            incumbent_tuple = list(zip(tis, tvs, strict=False))

        return incumbent_tuple
