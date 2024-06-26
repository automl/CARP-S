from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
import time
from multiprocessing import Pool
import pandas as pd

path = Path("carps/configs/problem/HPOBench/multifidelity")
config_fns = list(path.glob("*.yaml"))
result_file = "durations.csv"
# print(config_fns)
seed = 1

def measure_time(config_fn: Path, n: int = 5) -> float:
    cfg = OmegaConf.load(config_fn)
    cfg.problem.seed = seed
    problem = make_problem(cfg=cfg)
    config = problem.configspace.sample_configuration()
    trial_info = TrialInfo(config=config)
    durations = []
    for i in range(n):
        start = time.time()
        trial_value = problem.evaluate(trial_info)
        end = time.time()
        duration = end - start
        durations.append(duration)
    return np.mean(durations)

with Pool(processes=8) as pool:
    durations = pool.map(measure_time, config_fns)

df = pd.DataFrame({"config_fn": [str(p) for p in config_fns], "duration": durations})
df.to_csv("durations.csv", index=False)
