from __future__ import annotations

import os
from pathlib import Path

import pytest
from carps.utils.task import Task
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf


# Fixture to change the working directory to where pytest was called
@pytest.fixture(scope="session", autouse=False)
def _change_workdir():
    original_cwd = os.getcwd()  # Save the current working directory
    # Change to the directory where pytest was called
    os.chdir(Path(Path(__file__).parent).resolve())  # Change to the current test directory
    yield  # This allows the test to run while in the correct directory
    os.chdir(original_cwd)  # Restore the original working directory after tests


# Function to be tested (example)
def instantiate_task_and_evaluate(taskconfig_filename: str) -> bool:
    cfg = OmegaConf.load(taskconfig_filename)
    cfg.seed = 123
    task: Task = instantiate(cfg.task)
    task.objective_function.evaluate(TrialInfo(task.input_space.configuration_space.sample_configuration()))
    return True


# Fixture to collect filenames dynamically using glob
# @pytest.fixture(scope="module")
def filenames():
    # Replace with your desired pattern, e.g., "*.txt" to match all .txt files
    _filenames = list(Path("carps/configs/task").glob("**/*.yaml"))
    _filenames.sort()
    return _filenames


# Parametrize the test using the filenames
@pytest.mark.parametrize("filename", filenames())
def test_process_file(filename):
    result = instantiate_task_and_evaluate(filename)
    assert result  # Example assertion: Just check that the file isn't empty
