from omegaconf import OmegaConf
from carps.utils.running import make_problem

fns = [
    "carps/configs/problem/HPOBench/blackbox/surr/cfg_surr_ParamNet_Adult.yaml",
    "carps/configs/problem/HPOBench/blackbox/tab/cfg_ml_rf_7592.yaml",
    "carps/configs/problem/HPOBench/multifidelity/cfg_ml_lr_9981_subsample.yaml",
]

seed = 1
for fn in fns:
    cfg = OmegaConf.load(fn)
    
    cfg.problem.seed = seed
    if hasattr(cfg.problem, "problem"):
        cfg.problem.problem.rng = seed
    print(cfg)
    problem = make_problem(cfg=cfg)
    print(problem)
    del problem
