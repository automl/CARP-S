import os

from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from smacbenchmarking.wrappers.loggingproblemwrapper import LoggingProblemWrapper
from smacbenchmarking.container.containerized_problem_client import ContainerizedProblemClient
from smacbenchmarking.loggers.database_logger import DatabaseLogger
from smacbenchmarking.loggers.file_logger import FileLogger
from smacbenchmarking.utils.running import make_optimizer


def optimizer_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    loggers = [DatabaseLogger(result_processor), FileLogger()]
    problem = ContainerizedProblemClient(loggers)

    optimizer = make_optimizer(cfg=cfg, problem=problem)

    optimizer.run()


if (job_id := os.environ["BENCHMARKING_JOB_ID"]) != "":
    with open(f"{job_id}_pyexperimenter_id.txt", "r") as f:
        experiment_id = int(f.read())

    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = "smacbenchmarking/container/py_experimenter.yaml"

    if os.path.exists('smacbenchmarking/container/credentials.yaml'):
        database_credential_file = 'smacbenchmarking/container/credentials.yaml'
    else:
        database_credential_file = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="example_notebook",
        database_credential_file_path=database_credential_file,
        log_file=f"logs/{slurm_job_id}.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel
    )

    experimenter.unpause_experiment(experiment_id, optimizer_experiment)
