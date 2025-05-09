"""Run experiment from database via containers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor


def py_experimenter_evaluate(parameters: dict, result_processor: ResultProcessor, custom_config: dict) -> None:  # noqa: ARG001
    """Run one experiment from the database.

    Args:
        parameters (dict): Parameters from the database.
        result_processor (ResultProcessor): Result processor.
        custom_config (dict): Custom configuration.
    """
    config = parameters["config"]
    cfg_dict = json.loads(config)

    job_id = os.environ["BENCHMARKING_JOB_ID"]

    result_processor.process_results({"slurm_job_id": job_id})

    dict_config = OmegaConf.create(cfg_dict)
    cfg_path = f"{job_id}_hydra_config.yaml"
    OmegaConf.save(config=dict_config, f=cfg_path)

    with open(f"{job_id}_pyexperimenter_id.txt", "w+") as f:
        f.write(str(result_processor.experiment_id))

    with open(f"{job_id}_task_container.txt", "w+") as f:
        f.write(cfg_dict["benchmark_id"])

    with open(f"{job_id}_optimizer_container.txt", "w+") as f:
        f.write(cfg_dict["optimizer_container_id"])

    return ExperimentStatus.PAUSED


def main() -> None:
    """Excecute next experiment from PyExperimenter database."""
    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = "carps/experimenter/py_experimenter.yaml"

    if Path("carps/experimenter/credentials.yaml").exists():
        database_credential_file_path = "carps/experimenter/credentials.yaml"
    else:
        database_credential_file_path = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="example_notebook",
        database_credential_file_path=database_credential_file_path,
        log_file=f"logs/{slurm_job_id}.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


if __name__ == "__main__":
    main()
