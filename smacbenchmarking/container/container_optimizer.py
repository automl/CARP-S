import os

from hydra import initialize, compose

from smacbenchmarking.container.wrapper import ContainerizedProblemClient
from smacbenchmarking.run import make_optimizer


if (job_id := os.environ['BENCHMARKING_JOB_ID']) != '':
    with open(f"{job_id}_config.txt", 'r') as f:
        hydra_config_path = f.read()

    # path pattern runs/DUMMY_Optimizer/DUMMY/dummy/1/hydra_config.yaml
    # we divide into hydra_config.yaml and rest of path
    hydra_config_path = '../../' + hydra_config_path[:len(hydra_config_path) - 17]

    initialize(version_base=None, config_path=hydra_config_path)
    cfg = compose("hydra_config.yaml")

    problem = ContainerizedProblemClient()
    optimizer = make_optimizer(cfg=cfg, problem=problem)

