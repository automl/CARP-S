from __future__ import annotations

import os
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter

from carps.container.wrapper import ContainerizedProblemClient
from carps.loggers.database_logger import DatabaseLogger
from carps.loggers.file_logger import FileLogger
from carps.utils.running import make_optimizer

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor


def optimizer_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    loggers = [DatabaseLogger(result_processor), FileLogger()]
    problem = ContainerizedProblemClient(loggers=loggers)
    optimizer = make_optimizer(cfg=cfg, problem=problem)

    optimizer.run()


if (job_id := os.environ["BENCHMARKING_JOB_ID"]) != "":
    with open(f"{job_id}_pyexperimenter_id.txt") as f:
        experiment_id = int(f.read())

    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = "carps/container/py_experimenter.yaml"

    if os.path.exists("carps/container/credentials.yaml"):
        database_credential_file = "carps/container/credentials.yaml"
    else:
        database_credential_file = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="example_notebook",
        database_credential_file_path=database_credential_file,
        log_file=f"logs/{slurm_job_id}.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )

    experimenter.unpause_experiment(experiment_id, optimizer_experiment)
