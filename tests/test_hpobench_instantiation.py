# ml system singularity
from __future__ import annotations

from carps.utils.running import make_task
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf

fns = [
    "carps/configs/task/HPOBench/blackbox/surr/cfg_surr_ParamNet_Adult.yaml",
    "carps/configs/task/HPOBench/blackbox/surr/cfg_surr_SVM_default.yaml",
    "carps/configs/task/HPOBench/blackbox/tab/cfg_ml_rf_7592.yaml",
    "carps/configs/task/HPOBench/multifidelity/cfg_ml_lr_9981_subsample.yaml",
]

seed = 1
for fn in fns:
    cfg = OmegaConf.load(fn)

    cfg.task.seed = seed
    if hasattr(cfg.task, "task"):
        cfg.task.task.rng = seed
    print(cfg)
    task = make_task(cfg=cfg)
    print(task)

    config = task.objective_function.configspace.sample_configuration()
    res = [task.objective_function.evaluate(TrialInfo(config=config)).cost for _ in range(5)]
    print(res)
    del task
