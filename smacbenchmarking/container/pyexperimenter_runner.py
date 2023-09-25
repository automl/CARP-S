import ast
import json
import os

import hydra
from domdf_python_tools.utils import printr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor


def py_experimenter_evaluate(parameters: dict,
                             result_processor: ResultProcessor,
                             custom_config: dict):
    config = parameters['config']
    cfg_dict = ast.literal_eval(config)
    print(config)
    print(type(config))
    #cfg_dict = json.loads(config)

    printr(cfg_dict)

    job_id = 'test'
    #job_id = os.environ["BENCHMARKING_JOB_ID"]

    with open(f"{job_id}_problem_container.txt", 'w+') as f:
        f.write(cfg_dict["benchmark_id"])

    with open(f"{job_id}_optimizer_container.txt", 'w+') as f:
        f.write(cfg_dict["optimizer_id"])

    return None


def main() -> None:
    slurm_job_id = 'test'
    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='example_notebook',
                                  database_credential_file_path='01_lcbench_yahpo/credentials.cfg',
                                  log_file=f'logs/{slurm_job_id}.log')

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


if __name__ == "__main__":
    main()
