import os
import json
from inspect import getsourcefile
from os.path import abspath
import hydra
from pathlib import Path
import logging
from rich.logging import RichHandler
from smac.utils.logging import get_logger

from omegaconf import OmegaConf, DictConfig
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from carps.utils.running import optimize

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = get_logger("Run from DB")


def py_experimenter_evaluate(parameters: dict,
                             result_processor: ResultProcessor,
                             custom_config: dict):
    try:
        config = parameters['config']
        cfg_dict = json.loads(config)

        job_id = getattr(os.environ, "SLURM_JOB_ID", None)

        result_processor.process_results({"slurm_job_id": job_id})

        cfg = OmegaConf.create(cfg_dict)
        
        optimize(cfg, result_processor=result_processor)

        status = ExperimentStatus.DONE
    except Exception as e:
        print(e)
        status = ExperimentStatus.ERROR
        raise

    return status


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    slurm_job_id = getattr(os.environ, "SLURM_JOB_ID", None)

    experiment_configuration_file_path = cfg.pyexperimenter_configuration_file_path or Path(__file__).parent / "container/py_experimenter.yaml"

    database_credential_file_path = cfg.database_credential_file_path or Path(__file__).parent / "container/credentials.yaml"
    if database_credential_file_path is not None and not database_credential_file_path.exists():
        database_credential_file_path = None

    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name="example_notebook",
                                  database_credential_file_path=database_credential_file_path,
                                  log_file=f"logs/{slurm_job_id}.log",
                                  use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel
    )

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


if __name__ == "__main__":
    main()
