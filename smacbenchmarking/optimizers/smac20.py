from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from rich import print as printr

# from git import Repo
# from smac.callback.metadata_callback import MetadataCallback
from smac.facade.abstract_facade import AbstractFacade
from smac.runhistory.dataclasses import TrialInfo
from smac.scenario import Scenario

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer


class SMAC3Optimizer(Optimizer):
    def __init__(self, problem: Problem, smac_cfg: DictConfig) -> None:
        super().__init__(problem)

        self.configspace = self.problem.configspace
        self.smac_cfg = smac_cfg
        self._smac: AbstractFacade | None = None

    def convert_configspace(self, configspace: ConfigurationSpace) -> ConfigurationSpace:
        """Convert configuration space from Problem to Optimizer.

        Here, we don't need to convert.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns
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

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        trial_info = TrialInfo(config=config, seed=seed, budget=budget, instance=instance)

        return trial_info

    def target_function(
        self, config: Configuration, seed: int | None = None, budget: float | None = None, instance: str | None = None
    ) -> float | list[float]:
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
        float | list[float]
            Cost as float or list[float], depending on the number of objectives.
        """
        trial_info = self.convert_to_trial(config=config, seed=seed, budget=budget, instance=instance)
        trial_value = self.problem.evaluate(trial_info=trial_info)
        return trial_value.cost

    def setup_smac(self) -> AbstractFacade:
        """
        Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns
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

        if (
            smac_class == get_class("smac.facade.multi_fidelity_facade.MultiFidelityFacade")
            and "budget_variable" not in self.smac_cfg
        ):
            raise ValueError(
                "In order to use the MultiFidelityFacade you need to provide `budget_variable` at "
                "your configs root level indicating which variable of your config is the fidelity "
                "and controls the budget."
            )

        # Setup other SMAC kwargs
        smac_kwargs = {}
        if self.smac_cfg.smac_kwargs is not None:
            smac_kwargs = OmegaConf.to_container(self.smac_cfg.smac_kwargs, resolve=True, enum_to_str=True)

        # Instantiate Scenario
        scenario_kwargs = dict(
            configspace=self.configspace,
            # output_directory=Path(self.config.hydra.sweep.dir)
            # / "smac3_output",  # TODO document that output directory is automatically set
        )
        # We always expect scenario kwargs from the user
        _scenario_kwargs = OmegaConf.to_container(self.smac_cfg.scenario, resolve=True)
        scenario_kwargs.update(_scenario_kwargs)

        scenario = Scenario(**scenario_kwargs)

        smac_kwargs["scenario"] = scenario

        # If we have a custom intensifier we need to instantiate ourselves
        # because the helper methods in the facades expect a scenario.
        # Here it is easier to instantiate than completely via the yaml file.
        if "intensifier" in smac_kwargs and "intensifier_kwargs" in smac_kwargs:
            # Get, delete and update intensifier kwargs
            intensifier_kwargs = smac_kwargs["intensifier_kwargs"]
            del smac_kwargs["intensifier_kwargs"]
            intensifier_kwargs["scenario"] = scenario
            # Build intensifier
            intensifier = smac_kwargs["intensifier"]
            if isinstance(intensifier, str):
                intensifier = get_class(smac_kwargs["intensifier"])
            smac_kwargs["intensifier"] = intensifier(**intensifier_kwargs)

        if "initial_design" in smac_kwargs:
            initial_design = smac_kwargs["initial_design"]
            if isinstance(initial_design, str):
                initial_design = get_class(smac_kwargs["initial_design"])
            initial_design_kwargs = smac_kwargs.get("initial_design_kwargs", {})
            del smac_kwargs["initial_design_kwargs"]
            smac_kwargs["initial_design"] = initial_design(scenario=scenario, **initial_design_kwargs)

        printr(smac_class, smac_kwargs)

        smac = smac_class(
            target_function=self.target_function,
            **smac_kwargs,
        )

        return smac

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

        assert self._smac is not None
        rh = self._smac.runhistory
        trajectory = self._smac.intensifier.trajectory
        X: list[int | float] = []
        Y: list[float] = []

        for traj in trajectory:
            assert len(traj.config_ids) == 1
            config_id = traj.config_ids[0]
            config = rh.get_config(config_id)

            cost = rh.get_cost(config)
            if cost > 1e6:
                continue

            if sort_by == "trials":
                X.append(traj.trial)
            elif sort_by == "walltime":
                X.append(traj.walltime)
            else:
                raise RuntimeError("Unknown sort_by.")

            Y.append(cost)

        return X, Y

    def run(self) -> None:
        """Run SMAC on Problem.

        If SMAC is not instantiated, instantiate.
        """
        if self._smac is None:
            self._smac = self.setup_smac()

        incumbent = self._smac.optimize()  # noqa: F841

        return None
