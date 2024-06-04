from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import fire
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.requirements import check_requirements
from carps.utils.running import optimize

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor

setup_logging()
logger = get_logger("Run from DB")


def py_experimenter_evaluate(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    try:
        config = parameters["config"]
        cfg_dict = json.loads(config)

        job_id = getattr(os.environ, "SLURM_JOB_ID", None)
        if job_id is not None:
            result_processor.process_results({"slurm_job_id": job_id})

        cfg = OmegaConf.create(cfg_dict)

        check_requirements(cfg=cfg)

        # os.chdir(cfg.outdir)

        optimize(cfg, result_processor=result_processor)

        status = ExperimentStatus.DONE
    except Exception as e:
        print(e)
        status = ExperimentStatus.ERROR
        raise

    return status


def main(
    pyexperimenter_configuration_file_path: str | None = None, database_credential_file_path: str | None = None
) -> None:
    slurm_job_id = getattr(os.environ, "SLURM_JOB_ID", None)

    experiment_configuration_file_path = (
        pyexperimenter_configuration_file_path or Path(__file__).parent / "container/py_experimenter.yaml"
    )

    database_credential_file_path = (
        database_credential_file_path or Path(__file__).parent / "container/credentials.yaml"
    )
    if database_credential_file_path is not None and not database_credential_file_path.exists():
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
    fire.Fire(main)
