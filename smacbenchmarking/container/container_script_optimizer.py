import os
from configparser import ConfigParser

import sshtunnel
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from smacbenchmarking.benchmarks.loggingproblemwrapper import LoggingProblemWrapper
from smacbenchmarking.container.wrapper import ContainerizedProblemClient
from smacbenchmarking.loggers.database_logger import DatabaseLogger
from smacbenchmarking.loggers.file_logger import FileLogger
from smacbenchmarking.run import make_optimizer


def execute(slurm_job_id: int,
            experiment_configuration_file_path: str,
            experiment_id: int,
            optimizer_experiment: callable,
            database_credential_file: str = None
            ):
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='example_notebook',
                                  database_credential_file_path=database_credential_file,
                                  log_file=f'logs/{slurm_job_id}.log')

    experimenter.unpause_experiment(experiment_id, optimizer_experiment)


def optimizer_experiment(parameters: dict,
                         result_processor: ResultProcessor,
                         custom_config: dict):
    problem = ContainerizedProblemClient()
    logging_problem_wrapper = LoggingProblemWrapper(problem=problem)

    logging_problem_wrapper.add_logger(DatabaseLogger(result_processor))
    logging_problem_wrapper.add_logger(FileLogger())
    optimizer = make_optimizer(cfg=cfg, problem=logging_problem_wrapper)

    optimizer.run()


if (job_id := os.environ['BENCHMARKING_JOB_ID']) != '':
    with open(f"{job_id}_pyexperimenter_id.txt", 'r') as f:
        experiment_id = int(f.read())

    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'

    with open(experiment_configuration_file_path, 'r') as file:
        parsed_experiment_configuration_file = ConfigParser()
        parsed_experiment_configuration_file.read_file(file)

    if parsed_experiment_configuration_file['PY_EXPERIMENTER']['provider'] == 'mysql':
        database_credential_file = 'smacbenchmarking/container/credentials.cfg'
        configparser = ConfigParser()
        configparser.read_file(database_credential_file)
        config = configparser['TUNNEL_CONFIG']
        ssh_address_or_host = config['ssh_address_or_host']
        ssh_keypass = config['ssh_keypass']

        with sshtunnel.SSHTunnelForwarder(ssh_address_or_host=(ssh_address_or_host, 22),
                                          ssh_private_key_password=ssh_keypass,
                                          remote_bind_address=('127.0.0.1', 3306),
                                          local_bind_address=('127.0.0.1', 3306)
                                          ) as tunnel:
            execute(slurm_job_id, experiment_configuration_file_path, experiment_id, optimizer_experiment,
                    database_credential_file)

    else:
        execute(slurm_job_id, experiment_configuration_file_path, experiment_id, optimizer_experiment)
