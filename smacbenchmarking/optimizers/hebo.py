"""Implementation of HEBO Optimizer.

[2024-03-27]
Note that running `python smacbenchmarking/run.py +optimizer/hebo=config +problem/DUMMY=config seed=1 task.n_trials=25`
raises following error:
"linear_operator.utils.errors.NanError: cholesky_cpu: 4 of 4 elements of the torch.Size([2, 2]) tensor are NaN."

This is related to this issue: https://github.com/huawei-noah/HEBO/issues/61.

For non-dummy problems HEBO works fine.
"""
from __future__ import annotations

from collections import abc, OrderedDict

import pandas as pd
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
    IntegerHyperparameter,
    FloatHyperparameter,
)

from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer


def configspaceHP2HEBOHP(hp: Hyperparameter) -> dict:
    """Convert ConfigSpace hyperparameter to HEBO hyperparameter

    Parameters
    ----------
    hp : Hyperparameter
        ConfigSpace hyperparameter

    Returns
    -------
    dict
        HEBO hyperparameter

    Raises
    ------
    NotImplementedError
        If ConfigSpace hyperparameter is anything else than
        IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter,
        OrdinalHyperparameter or Constant
    """
    if isinstance(hp, IntegerHyperparameter):
        if hp.log:
            return {"name": hp.name, "type": "pow_int", "lb": hp.lower, "ub": hp.upper}
        else:
            return {"name": hp.name, "type": "int", "lb": hp.lower, "ub": hp.upper}
    elif isinstance(hp, FloatHyperparameter):
        if hp.log:
            return {"name": hp.name, "type": "pow", "lb": hp.lower, "ub": hp.upper}
        else:
            return {"name": hp.name, "type": "num", "lb": hp.lower, "ub": hp.upper}
    elif isinstance(hp, CategoricalHyperparameter):
        return {"name": hp.name, "type": "cat", "categories": hp.choices}
    elif isinstance(hp, OrdinalHyperparameter):
        return {
            "name": hp.name,
            "type": "step_int",
            "lb": 0,
            "ub": len(hp.sequence),
            "step": 1,
        }
    elif isinstance(hp, Constant):
        return {"name": hp.name, "type": "cat", "categories": [hp.value]}
    else:
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


def HEBOcfg2ConfigSpacecfg(
    hebo_suggestion: pd.DataFrame, design_space: DesignSpace, config_space: ConfigurationSpace
) -> Configuration:
    """Convert HEBO config to ConfigSpace config

    Parameters
    ----------
    hebo_suggestion : pd.DataFrame
        Configuration in HEBO format
    design_space : DesignSpace
        HEBO design space
    config_space : ConfigurationSpace
        ConfigSpace configuration space

    Returns
    -------
    Configuration
        Config in ConfigSpace format

    Raises
    ------
    ValueError
        If HEBO config is more than 1
    """
    if len(hebo_suggestion) > 1:
        raise ValueError(f"Only one suggestion is ok, got {len(hebo_suggestion)}.")
    hyp = hebo_suggestion.iloc[0].to_dict()
    for k in hyp:
        hp_type = design_space.paras[k]
        if hp_type.is_numeric and hp_type.is_discrete:
            hyp[k] = int(hyp[k])
            # Now we need to check if it is an ordinal hp
            hp_k = config_space.get_hyperparameter(k)
            if isinstance(hp_k, OrdinalHyperparameter):
                hyp[k] = hp_k.sequence[hyp[k]]

    return Configuration(configuration_space=config_space, values=hyp)


def ConfigSpacecfg2HEBOcfg(config: Configuration) -> pd.DataFrame:
    """Convert ConfigSpace config to HEBO suggestion

    Parameters
    ----------
    config : Configuration
        Configuration

    Returns
    -------
    pd.DataFrame
        Configuration in HEBO format, e.g.
            x1        x2
        0  2.817594  0.336420
    """
    config_dict = dict(config)
    rec = pd.DataFrame(config_dict, index=[0])
    return rec


class HEBOOptimizer(Optimizer):
    def __init__(
        self,
        problem: Problem,
        n_trials: int | None = None,
        time_budget: float | None = None,
        n_workers: int = 1,
        loggers: list[AbstractLogger] | None = None
    ) -> None:
        """
        Parameters
        ----------
        problem : Problem
            _description_
        num_trials : int | None, optional
            Number of trials after which the optimization is stopped, by default None
        max_wallclock_time : float | None, optional
            Time limit after which the optimization is stopped, by default None

        Raises
        ------
        ValueError
            If neither `num_trials` nor `max_wallclock_time` is specified.
        """
        super().__init__(problem, n_trials, time_budget, n_workers, loggers)

        # TODO: Extend HEBO to MO (maybe just adding a config suffices)
        self.configspace = self.problem.configspace

        self.hebo_configspace = self.convert_configspace(self.configspace)
        self.metric = getattr(problem, "metric", "cost")
        self.budget_type = getattr(self.problem, "budget_type", None)
        self.trial_counter = 0

        self._solver: HEBO | None = None

        self.completed_experiments: OrderedDict[int, tuple[TrialValue, TrialInfo]] = OrderedDict()

    def convert_configspace(self, configspace: ConfigurationSpace) -> DesignSpace:
        """Convert configuration space from Problem to Optimizer.

        Convert the configspace from ConfigSpace to HEBO. However, given that syne-tune does not support
        conditions and forbidden clauses, we only add hyperparameters here

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

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
        return ConfigSpacecfg2HEBOcfg(
            config=trial_info.config,
        )

    def convert_to_trial(self, rec: pd.DataFrame) -> TrialInfo:
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

        Returns
        -------
        TrialInfo
            trial info, needed to interact with the Problem
        """
        if len(rec) > 1:
            raise ValueError(f"Only one suggestion is ok, got {len(rec)}.")
        config = HEBOcfg2ConfigSpacecfg(
            hebo_suggestion=rec, design_space=self.hebo_configspace, config_space=self.problem.configspace
        )
        trial_info = TrialInfo(config=config, instance=None, budget=None, seed=None)
        return trial_info

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to run

        Returns
        -------
        TrialInfo
            Configuration, instance, seed, budget
        """
        rec = self._solver.suggest(1)
        trial_info = self.convert_to_trial(rec=rec)
        return trial_info

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell: Feed experiment results back to optimizer

        Parameters
        ----------
        trial_info : TrialInfo
            Configuration, instance, seed, budget
        trial_value : TrialValue
            Cost and additional information
        """
        cost = trial_value.cost
        suggestion = self.convert_from_trial(trial_info=trial_info)

        self.trial_counter += 1
        if isinstance(cost, abc.Sequence):
            cost = np.asarray([cost])
        else:
            cost = np.asarray(cost)

        self._solver.observe(suggestion, np.asarray([cost]))

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate target function.

        Store data point.

        Parameters
        ----------
        trial_info : TrialInfo
            Which config to evaluate (and instance, seed, budget)

        Returns
        -------
        TrialValue
            Information about function evaluation
        """
        trial_value = self.problem.evaluate(trial_info=trial_info)
        self.completed_experiments[self.trial_counter] = (trial_value, trial_info)
        return trial_value

    def _setup_optimizer(self) -> HEBO:
        """Setup Optimizer

        Returns
        -------
        HEBO
            Instance of a HEBO Optimizer

        """
        return HEBO(space=self.hebo_configspace)

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

        X: list[int | float] = []
        Y: list[float] = []

        current_incumbent = np.inf

        for k, v in self.completed_experiments.items():
            trial_value, trial_info = v
            cost = trial_value.cost
            if cost > 1e6:
                continue
            if current_incumbent < cost:
                current_incumbent = cost

                if sort_by == "trials":
                    X.append(k)
                elif sort_by == "walltime":
                    X.append(trial_value.endtime)
                else:
                    raise RuntimeError("Unknown sort_by.")

                Y.append(cost)

        return X, Y
    
    def current_incumbent(self) -> tuple[Configuration, np.ndarray | float] | list[tuple[Configuration, np.ndarray | float]] | None:
        best_x = self.solver.best_x
        best_y = self.solver.best_y
        config = HEBOcfg2ConfigSpacecfg(
            hebo_suggestion=best_x, design_space=self.hebo_configspace, config_space=self.problem.configspace
        )
        return (config, best_y)
