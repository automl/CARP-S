import ast
import os

from domdf_python_tools.utils import printr
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor


def py_experimenter_evaluate(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    config = parameters["config"]
    print(parameters)
    cfg_dict = ast.literal_eval(config)

    printr(cfg_dict)

    job_id = os.environ["BENCHMARKING_JOB_ID"]

    result_processor.process_results({"slurm_job_id": job_id})

    dict_config = OmegaConf.create(cfg_dict)
    cfg_path = f"{job_id}_hydra_config.yaml"
    OmegaConf.save(config=dict_config, f=cfg_path)

    with open(f"{job_id}_pyexperimenter_id.txt", "w+") as f:
        f.write(str(result_processor._experiment_id))

    with open(f"{job_id}_problem_container.txt", "w+") as f:
        f.write(cfg_dict["benchmark_id"])

    with open(f"{job_id}_optimizer_container.txt", "w+") as f:
        f.write(cfg_dict["optimizer_container_id"])

    return ExperimentStatus.PAUSED.value


def main() -> None:
    slurm_job_id = os.environ["BENCHMARKING_JOB_ID"]
    experiment_configuration_file_path = "smacbenchmarking/container/py_experimenter.cfg"
    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="example_notebook",
        database_credential_file_path="01_lcbench_yahpo/credentials.cfg",
        log_file=f"logs/{slurm_job_id}.log",
    )

    experimenter.execute(py_experimenter_evaluate, max_experiments=1)


if __name__ == "__main__":
    main()
