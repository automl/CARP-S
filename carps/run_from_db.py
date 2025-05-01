"""Run the optimization from the database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.requirements import check_requirements
from carps.utils.running import optimize

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor

setup_logging()
logger = get_logger("Run from DB")


def py_experimenter_evaluate(parameters: dict, result_processor: ResultProcessor, custom_config: dict) -> None:  # noqa: ARG001
    """Run one experiment from the database.

    Args:
        parameters (dict): Parameters from the database.
        result_processor (ResultProcessor): Result processor.
        custom_config (dict): Custom configuration.
    """
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


@hydra.main(config_path="configs", config_name="runfromdb.yaml", version_base=None)  # type: ignore[misc]
def main(
    cfg: DictConfig,
) -> None:
    """Run the optimization from the database.

    Connect to the database, pull one experiment and run this experiment.

    Usage: python run_from_db.py 'job_nr_dummy=range(1,1000)' -m
    This will create 1000 multirun jobs, each pulling an experiment from PyExperimenter and executing it.

    Args:
        cfg: DictConfig: Configuration file.
            pyexperimenter_configuration_file_path (str, optional): Path to the py_experimenter configuration file.
                Defaults to None.
            database_credential_file_path (str | Path, optional): Path to the database credential file.
                Defaults to None.
            experiment_name (str, optional): Name of the experiment. Defaults to "carps".
            job_nr_dummy: int | None: Dummy argument to create multirun jobs.
    """
    slurm_job_id = getattr(os.environ, "SLURM_JOB_ID", None)

    experiment_configuration_file_path = (
        cfg.pyexperimenter_configuration_file_path or Path(__file__).parent / "experimenter/py_experimenter.yaml"
    )

    database_credential_file_path = (
        cfg.database_credential_file_path or Path(__file__).parent / "experimenter/credentials.yaml"
    )
    database_credential_file_path = Path(database_credential_file_path)
    if database_credential_file_path is not None and not database_credential_file_path.exists():
        database_credential_file_path = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name=cfg.experiment_name,
        database_credential_file_path=database_credential_file_path,
        log_file=f"logs/{slurm_job_id}.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


if __name__ == "__main__":
    main()
