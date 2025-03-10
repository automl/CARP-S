from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from carps.utils.running import make_task
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf

path = Path("carps/configs/task/HPOBench/multifidelity")
config_fns = list(path.glob("*.yaml"))
result_file = "durations.csv"
# print(config_fns)
seed = 1

def measure_time(config_fn: Path, n: int = 5) -> float:
    cfg = OmegaConf.load(config_fn)
    cfg.task.seed = seed
    task = make_task(cfg=cfg)
    config = task.configspace.sample_configuration()
    trial_info = TrialInfo(config=config)
    durations = []
    for _i in range(n):
        start = time.time()
        task.objective_function.evaluate(trial_info)
        end = time.time()
        duration = end - start
        durations.append(duration)
    return np.mean(durations)

with Pool(processes=8) as pool:
    durations = pool.map(measure_time, config_fns)

df = pd.DataFrame({"config_fn": [str(p) for p in config_fns], "duration": durations})
df.to_csv("durations.csv", index=False)
