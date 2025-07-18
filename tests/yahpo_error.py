"""Test Yahpo in parallel.

Setup: Have a machine ready with 64 cores.
E.g., on a slurm cluster: salloc -t 01:00:00 -c 64
"""

from __future__ import annotations

from carps.objective_functions.yahpo import YahpoObjectiveFunction
from carps.utils.trials import TrialInfo
from joblib import Parallel, delayed


def make_yahpo_obj() -> YahpoObjectiveFunction:
    return YahpoObjectiveFunction(bench="lcbench", instance="167168", metric=["val_accuracy"], seed=0)


def init_and_sample(placeholder: int) -> None:  # noqa: ARG001
    obj_fun = make_yahpo_obj()
    config = obj_fun.configspace.sample_configuration()
    trial_info = TrialInfo(config=config)
    obj_fun.evaluate(trial_info=trial_info)


n_parallel = 64
Parallel(n_jobs=n_parallel)(delayed(init_and_sample)(i) for i in range(n_parallel))
