from __future__ import annotations

import time
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
    FloatHyperparameter
)

from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

from smacbenchmarking.utils.trials import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer


def configspaceHP2HEBOHP(hp: Hyperparameter) -> dict:
    if isinstance(hp, IntegerHyperparameter):
        if hp.log:
            return {'name': hp.name, 'type': 'pow_int', 'lb': hp.lower, 'ub': hp.upper}
        else:
            return {'name': hp.name, 'type': 'int', 'lb': hp.lower, 'ub': hp.upper}
    elif isinstance(hp, FloatHyperparameter):
        if hp.log:
            return {'name': hp.name, 'type': 'pow', 'lb': hp.lower, 'ub': hp.upper}
        else:
            return {'name': hp.name, 'type': 'num', 'lb': hp.lower, 'ub': hp.upper}
    elif isinstance(hp, CategoricalHyperparameter):
        return {'name': hp.name, 'type': 'cat', 'categories': hp.choices}
    elif isinstance(hp, OrdinalHyperparameter):
        return {'name': hp.name, 'type': 'step_int', 'lb': 0, 'ub': len(hp.sequence), 'step': 1, }
    elif isinstance(hp, Constant):
        return {'name': hp.name, 'type': 'cat', 'categories': [hp.value]}
    else:
        raise NotImplementedError(f'Unknown hyperparameter type: {hp.__class__.__name__}')


def HEBOcfg2ConfigSpacecfg(hebo_suggestion: pd.DataFrame,  design_space: DesignSpace,
                           config_space: ConfigurationSpace):
    # at the moment we only receive one suggestion from HEBO
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
    ...


class HEBOOptimizer(Optimizer):
    def __init__(self, problem: Problem,
                 max_budget: float | None = None,
                 num_trials: int | None = None,
                 wallclock_times: float | None = None) -> None:
        super().__init__(problem)
        self.fidelity_enabled = False

        self.configspace = self.problem.configspace

        self.hebo_configspace = self.convert_configspace(self.configspace)
        self.metric = getattr(problem, 'metric', 'cost')
        self.budget_type = getattr(self.problem, 'budget_type', None)
        self.trial_counter = 0
        self.max_budget = max_budget

        self._optimizer: HEBO | None = None

        if num_trials is None and wallclock_times is None:
            raise ValueError("either num_trials or wallclock_times must be given!")
        self.max_num_trials = num_trials
        self.start_time = time.time()
        self.wallclock_times = wallclock_times

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
        return ...

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
            hebo_suggestion=rec, 
            design_space=self.hebo_configspace, 
            config_space=self.problem.configspace
        )
        trial_info = TrialInfo(
            config=config,
            instance=None,
            budget=None,
            seed=None
        )
        return trial_info

    def ask(self) -> TrialInfo:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        rec = self._optimizer.suggest(1)
        trial_info = self.convert_to_trial(rec=rec)
        return trial_info

    def evaluate(self, trial_info: TrialInfo) -> float:
        cost = self.target_function(config=trial_info.config, instance=trial_info.instance, budget=trial_info.budget, seed=trial_info.seed)
        return cost

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """
        Feed experiment results back to the Scheduler

        :param suggestion: suggestions suggested by HEBO optimizer
        :param cost: float, cost values
        """
        cost = trial_value.cost
        suggestion = self.convert_from_trial(trial_info=trial_info)

        self.trial_counter += 1
        if isinstance(cost, abc.Sequence):
            cost = np.asarray([cost])
        else:
            cost = np.asarray(cost)
        self._optimizer.observe(suggestion, np.asarray([cost]))

    def target_function(
        self, config: Configuration, seed: int | None = None, budget: float | None = None, instance: str | None = None
    ) -> float:
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
        float
            cost
        """
        trial_info = TrialInfo(
            config=config,
            instance=instance,
            budget=budget,
            seed=seed
        )
        trial_value = self.problem.evaluate(trial_info=trial_info)
        if self.wallclock_times is not None:
            if trial_value.endtime - self.start_time > self.wallclock_times:
                # In this case, it is actually timed out. We will simply ignore that
                return trial_value.cost
        self.completed_experiments[self.trial_counter] = (trial_value, trial_info)
        return trial_value.cost

    def setup_optimizer(self) -> HEBO:
        """
        Setup Optimizer.

        Returns
        -------
        HEBO
            Instance of a HEBO Optimizer.

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

    def run(self) -> None:
        """Run HEBO on Problem.

        If HEBO is not instantiated, instantiate.
        """
        if self._optimizer is None:
            self._optimizer = self.setup_optimizer()
        self.start_time = time.time()
        while True:
            if self.max_num_trials is not None:
                if self.trial_counter >= self.max_num_trials:
                    break
            if self.wallclock_times is not None:
                if time.time() - self.start_time > self.wallclock_times:
                    break

            suggestion = self.ask()
            cost = self.evaluate(suggestion)
            self.tell(suggestion, cost)

        return None