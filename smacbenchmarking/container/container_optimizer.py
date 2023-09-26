import os

from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from smacbenchmarking.container.wrapper import ContainerizedProblemClient
from smacbenchmarking.loggers.database_logger import DatabaseLogger
from smacbenchmarking.loggers.file_logger import FileLogger
from smacbenchmarking.run import make_optimizer


def optimizer_experiment(parameters: dict,
                         result_processor: ResultProcessor,
                         custom_config: dict):
    problem = ContainerizedProblemClient()
    problem.add_logger(DatabaseLogger(problem, OmegaConf.create({}), result_processor))
    problem.add_logger(FileLogger(problem, OmegaConf.create({})))
    optimizer = make_optimizer(cfg=cfg, problem=problem)

    optimizer.run()


if (job_id := os.environ['BENCHMARKING_JOB_ID']) != '':
    with open(f"{job_id}_pyexperimenter_id.txt", 'r') as f:
        experiment_id = int(f.read())

    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='example_notebook',
                                  database_credential_file_path='01_lcbench_yahpo/credentials.cfg',
                                  log_file=f'logs/{slurm_job_id}.log')

    experimenter.unpause_experiment(experiment_id, optimizer_experiment)
