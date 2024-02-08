import ast
import os
from configparser import ConfigParser

import sshtunnel
from domdf_python_tools.utils import printr
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor


def py_experimenter_evaluate(parameters: dict,
                             result_processor: ResultProcessor,
                             custom_config: dict):
    config = parameters['config']
    print(parameters)
    cfg_dict = ast.literal_eval(config)

    printr(cfg_dict)

    job_id = os.environ["BENCHMARKING_JOB_ID"]

    result_processor.process_results({'slurm_job_id': job_id})

    dict_config = OmegaConf.create(cfg_dict)
    cfg_path = f"{job_id}_hydra_config.yaml"
    OmegaConf.save(config=dict_config, f=cfg_path)

    with open(f"{job_id}_pyexperimenter_id.txt", 'w+') as f:
        f.write(str(result_processor._experiment_id))

    with open(f"{job_id}_problem_container.txt", 'w+') as f:
        f.write(cfg_dict["benchmark_id"])

    with open(f"{job_id}_optimizer_container.txt", 'w+') as f:
        f.write(cfg_dict["optimizer_id"])

    return ExperimentStatus.PAUSED.value


def execute(experiment_configuration_file_path: str,
            slurm_job_id: str,
            database_credential_file_path: str = None,
            ):
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='example_notebook',
                                  database_credential_file_path=database_credential_file_path,
                                  log_file=f'logs/{slurm_job_id}.log')

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


def main() -> None:
    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'

    with open(experiment_configuration_file_path, 'r') as file:
        parsed_experiment_configuration_file = ConfigParser()
        parsed_experiment_configuration_file.read_file(file)

    if parsed_experiment_configuration_file['PY_EXPERIMENTER']['provider'] == 'mysql':
        database_credential_file_path = 'smacbenchmarking/container/credentials.cfg'
        with open(database_credential_file_path, 'r') as file:
            configparser = ConfigParser()
            configparser.read_file(file)
            config = configparser['TUNNEL_CONFIG']
            ssh_address_or_host = config['ssh_address_or_host']
            ssh_keypass = config['ssh_keypass']

        with sshtunnel.SSHTunnelForwarder(ssh_address_or_host=(ssh_address_or_host, 22),
                                          ssh_private_key_password=ssh_keypass,
                                          remote_bind_address=('127.0.0.1', 3306),
                                          local_bind_address=('127.0.0.1', 3306)
                                          ) as tunnel:
            execute(experiment_configuration_file_path, slurm_job_id,
                    database_credential_file_path)

    else:
        execute(experiment_configuration_file_path, slurm_job_id)


if __name__ == "__main__":
    main()
