from __future__ import annotations

from multiprocessing import Pool

from carps.objective_functions.yahpo import YahpoObjectiveFunction
from carps.utils.trials import TrialInfo


def make_yahpo_obj() -> YahpoObjectiveFunction:
    return YahpoObjectiveFunction(bench="lcbench", instance="167168", metric=["val_accuracy"], seed=0)


def init_and_sample(placeholder: int) -> None:  # noqa: ARG001
    obj_fun = make_yahpo_obj()
    config = obj_fun.configspace.sample_configuration()
    trial_info = TrialInfo(config=config)
    obj_fun.evaluate(trial_info=trial_info)


obj_fun = make_yahpo_obj()
n_parallel = 32
with Pool(n_parallel) as pool:
    pool.map(init_and_sample, range(n_parallel))
