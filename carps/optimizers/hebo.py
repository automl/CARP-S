"""Implementation of HEBO Optimizer.

[2024-03-27]
Note that running `python carps/run.py +optimizer/hebo=config +task/DUMMY=config seed=1
 task.optimization_resources.n_trials=25`
raises following error:
"linear_operator.utils.errors.NanError: cholesky_cpu: 4 of 4 elements of the torch.Size([2, 2]) tensor are NaN."

This is related to this issue: https://github.com/huawei-noah/HEBO/issues/61.

For non-dummy objective functions HEBO works fine.
"""

from __future__ import annotations

from collections import OrderedDict, abc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from carps.optimizers.optimizer import Optimizer
from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent

setup_logging()
logger = get_logger("HEBO")

SMALL_NUMBER = 1e6


def configspaceHP2HEBOHP(hp: Hyperparameter) -> dict:  # noqa: PLR0911, N802
    """Convert ConfigSpace hyperparameter to HEBO hyperparameter.

    Parameters
    ----------
    hp : Hyperparameter
        ConfigSpace hyperparameter

    Returns:
    -------
    dict
        HEBO hyperparameter

    Raises:
    ------
    NotImplementedError
        If ConfigSpace hyperparameter is anything else than
        IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter,
        OrdinalHyperparameter or Constant
    """
    if isinstance(hp, IntegerHyperparameter):
        if hp.log:
            return {"name": hp.name, "type": "pow_int", "lb": hp.lower, "ub": hp.upper}
        return {"name": hp.name, "type": "int", "lb": hp.lower, "ub": hp.upper}
    if isinstance(hp, FloatHyperparameter):
        if hp.log:
            return {"name": hp.name, "type": "pow", "lb": hp.lower, "ub": hp.upper}
        return {"name": hp.name, "type": "num", "lb": hp.lower, "ub": hp.upper}
    if isinstance(hp, CategoricalHyperparameter):
        return {"name": hp.name, "type": "cat", "categories": hp.choices}
    if isinstance(hp, OrdinalHyperparameter):
        # In HEBO, ordinal hyperparameters are treated as integer hyperparameters.
        # The index of the sequence is used as the value and therefore must be
        # 0-based.
        return {
            "name": hp.name,
            "type": "step_int",
            "lb": 0,
            "ub": len(hp.sequence) - 1,
            "step": 1,
        }
    if isinstance(hp, Constant):
        return {"name": hp.name, "type": "cat", "categories": [hp.value]}

    raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


def HEBOcfg2ConfigSpacecfg(  # noqa: N802
    hebo_suggestion: pd.DataFrame,
    design_space: DesignSpace,
    config_space: ConfigurationSpace,
    allow_inactive_with_values: bool = True,  # noqa: FBT001, FBT002
) -> Configuration:
    """Convert HEBO config to ConfigSpace config.

    Parameters
    ----------
    hebo_suggestion : pd.DataFrame
        Configuration in HEBO format
    design_space : DesignSpace
        HEBO design space
    config_space : ConfigurationSpace
        ConfigSpace configuration space
    allow_inactive_with_values : bool
        Allow values for inactive hyperparameters. This is relevant if the space has
        conditionals but the optimizer does not support them.

    Returns:
    -------
    Configuration
        Config in ConfigSpace format

    Raises:
    ------
    ValueError
        If HEBO config is more than 1
    """
    if len(hebo_suggestion) > 1:
        raise ValueError(f"Only one suggestion is ok, got {len(hebo_suggestion)}.")
    hyp = hebo_suggestion.iloc[0].to_dict()
    for k in hyp:
        hp_type = design_space.paras[k]
        if hp_type.is_numeric and hp_type.is_discrete and not np.isnan(hyp[k]):
            hyp[k] = int(hyp[k])
            # Now we need to check if it is an ordinal hp
            hp_k = config_space[k]
            if isinstance(hp_k, OrdinalHyperparameter):
                hyp[k] = hp_k.sequence[hyp[k]]

    return Configuration(
        configuration_space=config_space, values=hyp, allow_inactive_with_values=allow_inactive_with_values
    )


def ConfigSpacecfg2HEBOcfg(config: Configuration) -> pd.DataFrame:  # noqa: N802
    """Convert ConfigSpace config to HEBO suggestion.

    Parameters
    ----------
    config : Configuration
        Configuration

    Returns:
    -------
    pd.DataFrame
        Configuration in HEBO format, e.g.
            x1        x2
        0  2.817594  0.336420
    """
    config_dict = dict(config)
    print(config_dict)
    return pd.DataFrame(config_dict, index=[0])


class HEBOOptimizer(Optimizer):
    """HEBO Optimizer."""

    def __init__(
        self,
        task: Task,
        hebo_cfg: DictConfig | None = None,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Interface to HEBO (https://github.com/huawei-noah/HEBO) [1].

        [1] Cowen-Rivers, Alexander I., et al. "An Empirical Study of Assumptions in Bayesian Optimisation."
            arXiv preprint arXiv:2012.03826 (2021).

        HEBO does not support conditional configuration spaces as well as priors for hyperparameters.


        Parameters
        ----------
        task : Task
            The task (objective function with specific input and output space and optimization resources) to optimize.
        hebo_cfg : DictConfig, optional
            Optional kwargs for HEBO class.
        loggers : list[AbstractLogger], optional
            List of loggers to use, by default None.
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

        # TODO: Extend HEBO to MO (maybe just adding a config suffices)
        self.configspace = self.task.input_space.configuration_space

        if len(self.configspace.conditions) > 0:
            logger.warning(
                "HEBO treats search spaces with conditions as spaces without, "
                "as it does not support conditional search spaces."
            )
            # msg = "HEBO does not support conditional search spaces."
            # raise RuntimeError(msg)

        self.hebo_configspace = self.convert_configspace(self.configspace)
        self.budget_type = getattr(self.task.input_space.fidelity_space, "fidelity_type", None)
        self.trial_counter = 0
        hebo_cfg = {} if hebo_cfg is None else dict(hebo_cfg)
        self.hebo_cfg = hebo_cfg

        self._solver: HEBO | None = None

        self.completed_experiments: OrderedDict[str, tuple[TrialValue, TrialInfo]] = OrderedDict()

    def convert_configspace(self, configspace: ConfigurationSpace) -> DesignSpace:
        """Convert configuration space from ObjectiveFunction to Optimizer.

        Convert the configspace from ConfigSpace to HEBO. However, given that syne-tune does not support
        conditions and forbidden clauses, we only add hyperparameters here

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from ObjectiveFunction.

        dict[str, Any]
        -------
        DesignSpace
            HEBO design space
        """
        hps_hebo = []
        for _, v in configspace.items():
            hps_hebo.append(configspaceHP2HEBOHP(v))
        return DesignSpace().parse(hps_hebo)

    def convert_from_trial(self, trial_info: TrialInfo) -> pd.DataFrame:
        """Convert from trial info to HEBO's recommendation.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info containing configuration.

        Returns:
        -------
        pd.DataFrame
            HEBO recommendation.
        """
        return ConfigSpacecfg2HEBOcfg(
            config=trial_info.config,
        )

    def convert_to_trial(self, rec: pd.DataFrame) -> TrialInfo:  # type: ignore[override]
        """Convert HEBO's recommendation to trial info.

        Parameters
        ----------
        rec : pd.DataFrame
            HEBO recommendation, can look like this for 2d:
                x1        x2
            0  2.817594  0.336420
            1 -2.293059 -1.381435
            2 -0.666595  2.016661
            3  0.130466 -3.203030

            These are four points.

        Returns:
        -------
        TrialInfo
            trial info, needed to interact with the ObjectiveFunction
        """
        if len(rec) > 1:
            raise ValueError(f"Only one suggestion is ok, got {len(rec)}.")
        config = HEBOcfg2ConfigSpacecfg(
            hebo_suggestion=rec,
            design_space=self.hebo_configspace,
            config_space=self.task.input_space.configuration_space,
            allow_inactive_with_values=True,
        )
        return TrialInfo(config=config, instance=None, budget=None, seed=None)

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to run.

        Returns:
        -------
        TrialInfo
            Configuration, instance, seed, budget
        """
        assert self._solver is not None
        rec = self._solver.suggest(1)
        return self.convert_to_trial(rec=rec)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell: Feed experiment results back to optimizer.

        Parameters
        ----------
        trial_info : TrialInfo
            Configuration, instance, seed, budget
        trial_value : TrialValue
            Cost and additional information
        """
        assert self._solver is not None
        cost = trial_value.cost
        suggestion = self.convert_from_trial(trial_info=trial_info)

        cost = np.asarray([cost]) if isinstance(cost, abc.Sequence) else np.asarray(cost)

        self._solver.observe(suggestion, np.asarray([cost]))

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate target function.

        Store data point.

        Parameters
        ----------
        trial_info : TrialInfo
            Which config to evaluate (and instance, seed, budget)

        Returns:
        -------
        TrialValue
            Information about function evaluation
        """
        trial_value = self.task.objective_function.evaluate(trial_info=trial_info)
        self.completed_experiments[str(self.trial_counter)] = (trial_value, trial_info)
        return trial_value

    def _setup_optimizer(self) -> HEBO:
        """Setup Optimizer.

        Returns:
        -------
        HEBO
            Instance of a HEBO Optimizer

        """
        return HEBO(space=self.hebo_configspace, **self.hebo_cfg)

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        """List of x and y values of the incumbents over time. x depends on ``sort_by``.

        Parameters
        ----------
        sort_by: str
            Can be "trials" or "walltime".

        Returns:
        -------
        tuple[list[float], list[float]]

        """
        # if len(self.task.output_space.objectives) > 1:
        #     raise NotSupportedError

        X: list[int | float] = []
        Y: list[float] = []

        current_incumbent = np.inf

        for k, v in self.completed_experiments.items():
            trial_value, trial_info = v
            cost = trial_value.cost
            assert isinstance(cost, float)  # HEBO does not support multi-objective optimization
            if cost > SMALL_NUMBER:
                continue
            if current_incumbent < cost:
                current_incumbent = cost

                if sort_by == "trials":
                    X.append(float(k))
                elif sort_by == "walltime":
                    X.append(trial_value.endtime)
                else:
                    raise RuntimeError("Unknown sort_by.")

                Y.append(cost)

        return X, Y

    def get_current_incumbent(self) -> Incumbent:
        """Get the current incumbent.

        Returns:
        -------
        Incumbent
            The best configuration so far and its cost as a tuple
            of (TrialInfo, TrialValue).
        """
        best_x = self.solver.best_x
        best_y = self.solver.best_y
        config = HEBOcfg2ConfigSpacecfg(
            hebo_suggestion=best_x,
            design_space=self.hebo_configspace,
            config_space=self.task.input_space.configuration_space,
            allow_inactive_with_values=True,
        )
        trial_info = TrialInfo(config=config)
        trial_value = TrialValue(cost=best_y)
        return (trial_info, trial_value)
