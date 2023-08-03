import os

from hydra import initialize, compose

from smacbenchmarking.container.wrapper import ContainerizedProblemClient
from smacbenchmarking.run import make_optimizer


if (job_id := os.environ['BENCHMARKING_JOB_ID']) != '':
    with open(f"{job_id}_config.txt", 'r') as f:
        hydra_config_path = f.read()

    initialize(version_base=None, config_path="conf")
    cfg = compose(hydra_config_path)

    problem = ContainerizedProblemClient()
    optimizer = make_optimizer(cfg=cfg, problem=problem)

