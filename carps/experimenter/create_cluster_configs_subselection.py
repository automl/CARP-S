"""Fill database with subselection experiments. Not the most elegant script."""

from __future__ import annotations

import itertools
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from hydra.core.utils import setup_globals
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter
from tqdm import tqdm

from carps.experimenter.create_cluster_configs import experiment_identifiers, get_experiment_definition

setup_globals()


def get_all_config_files_in_path(path: Path) -> list[Path]:
    """Get a list of all files matching the glob pattern.

    Args:
        path (Path): Path to search in.

    Returns:
        list[Path]: List of all yaml files.
    """
    return list(path.glob("*.yaml"))


def get_all_config_files_in_paths(paths: list[str | Path]) -> list[Path]:
    """Get a list of all files matching the glob pattern.

    Args:
        paths: list[str|Path]: Paths to search in.

    Returns:
        list[Path]: List of all yaml files.
    """
    all_files = []
    for path in paths:
        files = get_all_config_files_in_path(Path(path))
        print(f"Found {len(files)} tasks in {path}")
        all_files.extend(files)
    return all_files


def create_combinations(
    optimizer_fns: list[Path], task_fns: list[Path], seeds: list[int]
) -> list[tuple[Path, Path, int]]:
    """Create all combinations of optimizers, tasks, and seeds.

    Args:
        optimizer_fns (list[Path]): List of optimizer config files.
        task_fns (list[Path]): List of task config files.
        seeds (list[int]): List of seeds.

    Returns:
        list[tuple[Path, Path, int]]: List of tuples containing optimizer, task, and seed.
    """
    return list(itertools.product(optimizer_fns, task_fns, seeds))


def load_combination(optimizer_config_fn: Path, task_config_fn: Path, seed: int) -> DictConfig:
    """Load a combination of optimizer, task, and seed.

    Args:
        optimizer_config_fn (Path): Path to the optimizer config file.
        task_config_fn (Path): Path to the task config file.
        seed (int): Seed.

    Returns:
        DictConfig: Merged configuration.
    """
    base_cfg = OmegaConf.load("../carps/configs/base.yaml")

    optimizer_config_fn = Path(optimizer_config_fn)
    optimizer_config_base = None
    if "base" in optimizer_config_fn.read_text():
        # If the optimizer config is a base config, load the base config and the task config
        optimizer_config_fn_base = Path(optimizer_config_fn).parent / "base.yaml"
        optimizer_config_base = OmegaConf.load(optimizer_config_fn_base)
    # Load the optimizer config
    optimizer_config = OmegaConf.load(optimizer_config_fn)

    if optimizer_config_base:
        optimizer_config = OmegaConf.merge(optimizer_config_base, optimizer_config)
    # Load the task config
    task_config = OmegaConf.load(task_config_fn)
    # Create a new config by merging the two configs
    config = OmegaConf.merge(base_cfg, optimizer_config, task_config)
    # Set the seed
    config.seed = int(seed)
    return config


def load(combo: tuple) -> dict:
    """Load experiment definiton.

    Args:
        combo (tuple): Tuple containing optimizer config filename, task config filename, and seed.

    Returns:
        dict: Experiment definition.
    """
    cfg = load_combination(*combo)
    return get_experiment_definition(cfg)


def check_existance_by_keys(experiment_definition: dict, existing_rows: list, identifier_keys: list[str]) -> bool:
    """Check existance of experiment in database by the identifier keys.

    Args:
        experiment_definition (dict): Experiment definition.
        existing_rows (list): List of existing rows in the database.
        identifier_keys (list[str]): List of keys to check for existance.

    Returns:
        bool: True if the experiment exists, False otherwise.
    """
    return any(all(experiment_definition[k] == e[k] for k in identifier_keys) for e in existing_rows)


# OPTIMIZERS
bb_optimizers = (
    "../carps/configs/optimizer/randomsearch/config.yaml",
    "../carps/configs/optimizer/Ax/config.yaml",
    "../carps/configs/optimizer/hebo/config.yaml",
    "../carps/configs/optimizer/nevergrad/bayesopt.yaml",
    "../carps/configs/optimizer/nevergrad/Hyperopt.yaml",
    "../carps/configs/optimizer/nevergrad/NoisyBandit.yaml",
    "../carps/configs/optimizer/nevergrad/DE.yaml",
    "../carps/configs/optimizer/nevergrad/ES.yaml",
    "../carps/configs/optimizer/optuna/SO_TPE.yaml",
    "../carps/configs/optimizer/smac20/blackbox.yaml",
    "../carps/configs/optimizer/smac20/hpo.yaml",
    "../carps/configs/optimizer/synetune/BO.yaml",
    "../carps/configs/optimizer/synetune/BORE.yaml",
    "../carps/configs/optimizer/synetune/KDE.yaml",
    "../carps/configs/optimizer/synetune/MOREA.yaml",
    "../carps/configs/optimizer/scikit_optimize/BO_GP_EI.yaml",
    "../carps/configs/optimizer/scikit_optimize/BO_GP_LCB.yaml",
    "../carps/configs/optimizer/scikit_optimize/BO_GP_PI.yaml",
    "../carps/configs/optimizer/scikit_optimize/BO.yaml",
)

mf_optimzier = (
    "../carps/configs/optimizer/randomsearch/config.yaml",
    "../carps/configs/optimizer/dehb/multifidelity.yaml",
    "../carps/configs/optimizer/smac20/hyperband.yaml",
    "../carps/configs/optimizer/smac20/multifidelity.yaml",
    "../carps/configs/optimizer/synetune/DEHB.yaml",
    "../carps/configs/optimizer/synetune/SyncMOBSTER.yaml",
)

mo_optimizers = (
    "../carps/configs/optimizer/randomsearch/config.yaml",
    "../carps/configs/optimizer/nevergrad/cmaes.yaml",
    "../carps/configs/optimizer/nevergrad/DE.yaml",
    "../carps/configs/optimizer/nevergrad/ES.yaml",
    "../carps/configs/optimizer/optuna/MO_NSGAII.yaml",
    "../carps/configs/optimizer/optuna/MO_TPE.yaml",
    "../carps/configs/optimizer/smac20/multiobjective_gp.yaml",
    "../carps/configs/optimizer/smac20/multiobjective_rf.yaml",
    "../carps/configs/optimizer/synetune/BO_MO_LS.yaml",
    "../carps/configs/optimizer/synetune/BO_MO_RS.yaml",
    "../carps/configs/optimizer/synetune/MOREA.yaml",
)

momf_optimizers = (
    "../carps/configs/optimizer/randomsearch/config.yaml",
    "../carps/configs/optimizer/smac20/momf_gp.yaml",
    "../carps/configs/optimizer/smac20/momf_rf.yaml",
    "../carps/configs/optimizer/nevergrad/cmaes.yaml",
)

# TASKS
bb_task_paths = [
    "../carps/configs/task/subselection/blackbox/dev",
    "../carps/configs/task/subselection/blackbox/test",
]
mf_task_paths = [
    "../carps/configs/task/subselection/multifidelity/dev",
    "../carps/configs/task/subselection/multifidelity/test",
]
mo_task_paths = [
    "../carps/configs/task/subselection/multiobjective/dev",
    "../carps/configs/task/subselection/multiobjective/test",
]
momf_task_paths = [
    "../carps/configs/task/subselection/multifidelityobjective/dev",
    "../carps/configs/task/subselection/multifidelityobjective/test",
]

bb_tasks = get_all_config_files_in_paths(bb_task_paths)  # type:ignore[arg-type]
mf_tasks = get_all_config_files_in_paths(mf_task_paths)  # type:ignore[arg-type]
mo_tasks = get_all_config_files_in_paths(mo_task_paths)  # type:ignore[arg-type]
momf_tasks = get_all_config_files_in_paths(momf_task_paths)  # type:ignore[arg-type]

# SEEDS
seeds = np.arange(1, 21)

# CREATE COMBINATIONS
combinations_bb = create_combinations(bb_optimizers, bb_tasks, seeds)  # type:ignore[arg-type]
combinations_mf = create_combinations(mf_optimzier, mf_tasks, seeds)  # type:ignore[arg-type]
combinations_mo = create_combinations(mo_optimizers, mo_tasks, seeds)  # type:ignore[arg-type]
combinations_momf = create_combinations(momf_optimizers, momf_tasks, seeds)  # type:ignore[arg-type]
print(f"len bb combinations: {len(combinations_bb)}")
print(f"len mf combinations: {len(combinations_mf)}")
print(f"len mo combinations: {len(combinations_mo)}")
print(f"len momf combinations: {len(combinations_momf)}")
combinations = combinations_bb + combinations_mf + combinations_mo + combinations_momf

# CREATE EXPERIMENT DEFINITIONS
num_processes = 8
pool = Pool(processes=num_processes)
exp_defs = list(tqdm(pool.imap_unordered(load, combinations), total=len(combinations)))
pool.close()
pool.join()

# CONNEC TO DATABASE and get existing experiments
experiment_configuration_file_path = Path(__file__).parent / "py_experimenter.yaml"
database_credential_file_path = Path(__file__).parent / "credentials.yaml"
if database_credential_file_path is not None and not database_credential_file_path.exists():
    database_credential_file_path = None  # type: ignore[assignment]

experimenter = PyExperimenter(
    experiment_configuration_file_path=experiment_configuration_file_path,
    name="carps",
    database_credential_file_path=database_credential_file_path,
    log_level=logging.INFO,
    use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
)
column_names = list(experimenter.db_connector.database_configuration.keyfields.keys())
existing_rows = experimenter.db_connector._get_existing_rows(column_names)

# Check if experiments exists
rows_exist = [
    check_existance_by_keys(exp_def, existing_rows, experiment_identifiers)
    for exp_def in tqdm(exp_defs, total=len(exp_defs))
]
print(f"This number of experiments already exists: {np.sum(rows_exist)}")

experiments_to_add = [exp_def for exp_def, exists in zip(exp_defs, rows_exist, strict=True) if not exists]
print(
    f"number of existing rows {len(existing_rows)}, previous length: "
    f"{len(exp_defs)}, length now {len(experiments_to_add)}"
)
experimenter.fill_table_with_rows(experiments_to_add)
