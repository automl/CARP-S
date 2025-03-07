"""Test optimizers."""

from __future__ import annotations

from pathlib import Path

import pytest
from carps.utils.task import Task
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

SEED = 123


# Function to be tested (example)
def instantiate_task_and_evaluate(taskconfig_filename: str) -> bool:
    cfg = OmegaConf.load(taskconfig_filename)
    cfg.seed = SEED
    task: Task = instantiate(cfg.task)
    task.objective_function.evaluate(TrialInfo(task.input_space.configuration_space.sample_configuration()))
    return True


def filenames():
    # Replace with your desired pattern, e.g., "*.txt" to match all .txt files
    _filenames = list(Path("carps/configs/optimizer").glob("**/*.yaml"))
    _filenames = [fn for fn in _filenames if "base" not in str(fn)]
    _filenames = [fn for fn in _filenames if "hebo" not in str(fn)]
    _filenames.sort()
    return _filenames


def get_task_cfg(optimizer_cfg: DictConfig) -> DictConfig:
    # MOMF
    if optimizer_cfg.get("expects_multiple_objectives", False) and optimizer_cfg.get("expects_fidelities", False):
        return OmegaConf.load("carps/configs/task/DUMMY/momf.yaml")
    # MO
    if optimizer_cfg.get("expects_multiple_objectives", False):
        return OmegaConf.load("carps/configs/task/DUMMY/multiobjective.yaml")
    # MF
    if optimizer_cfg.get("expects_fidelities", False):
        return OmegaConf.load("carps/configs/task/DUMMY/multifidelity.yaml")
    # Single-objective, single-fidelity
    return OmegaConf.load("carps/configs/task/DUMMY/config.yaml")


def add_resources(task_cfg: DictConfig) -> DictConfig:
    task_cfg.task.optimization_resources.n_trials = 3
    task_cfg.seed = SEED
    return task_cfg


# Parametrize the test using the filenames
@pytest.mark.parametrize("optimizer_filename", filenames())
def test_run_optimizer(optimizer_filename, tmpdir):
    # Load optimizer configuration
    cfg = OmegaConf.load(optimizer_filename)

    # Set missing interpolation keys
    cfg.seed = SEED
    cfg.outdir = str(tmpdir)

    # Get task configuration
    task_cfg = get_task_cfg(cfg.optimizer)
    task_cfg = add_resources(task_cfg)

    # Instantiate task
    task = instantiate(task_cfg.task)

    # Merge task configuration with optimizer configuration and load default optimizer config if necessary
    cfg = OmegaConf.merge(cfg, task_cfg)
    defaults = cfg.get("defaults", None)
    if defaults is not None:
        if list(cfg.defaults) == ["base"]:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(optimizer_filename).parent / "base.yaml"))
        else:
            raise ValueError(f"Unknown defaults: {cfg.defaults}. Cannot pars {optimizer_filename} like this.")

    # Instantiate optimizer
    optimizer = instantiate(cfg.optimizer)(task=task)

    # Run optimizer
    optimizer.run()
