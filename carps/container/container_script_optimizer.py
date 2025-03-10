"""Container script for the optimizer container."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter

from carps.container.wrapper import ContainerizedTaskClient
from carps.loggers.database_logger import DatabaseLogger
from carps.loggers.file_logger import FileLogger
from carps.utils.running import make_optimizer

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor


def optimizer_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict) -> None:  # noqa: ARG001
    """Perform one optimization run with the experiment parameters from the PyExperimenter database.

    Parameters
    ----------
    parameters : dict
        The parameters of the experiment.
    result_processor : ResultProcessor
        The result processor.
    """
    loggers = [DatabaseLogger(result_processor), FileLogger()]
    task = ContainerizedTaskClient(loggers=loggers)
    optimizer = make_optimizer(cfg=cfg, task=task)  # type: ignore[arg-type]

    optimizer.run()


# Execute this when the environment variable is set
if (job_id := os.environ["BENCHMARKING_JOB_ID"]) != "":
    with open(f"{job_id}_pyexperimenter_id.txt") as f:
        experiment_id = int(f.read())

    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = "carps/container/py_experimenter.yaml"

    kwargs = {}
    if Path("carps/container/credentials.yaml").exists():
        database_credential_file = "carps/container/credentials.yaml"
        kwargs["database_credential_file"] = database_credential_file

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="example_notebook",
        log_file=f"logs/{slurm_job_id}.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
        **kwargs,
    )

    experimenter.unpause_experiment(experiment_id, optimizer_experiment)
