from __future__ import annotations
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from pathlib import Path
import logging
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle as pckl
from hydra.core.utils import setup_globals


setup_globals()


experiment_identifiers = ["optimizer_id", "task_id", "seed", "benchmark_id", "n_trials", "time_budget"]


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



folder_path = Path("configs_pckl")
pkl_files = list(folder_path.glob("*.pkl"))
print('length of pkl_files', len(pkl_files))

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pckl.load(f)

with ThreadPoolExecutor() as executor:
    exp_defs = list(executor.map(load_pickle, pkl_files))


# CONNECT TO DATABASE and get existing experiments
experiment_configuration_file_path = "carps/experimenter/py_experimenter copy.yaml"
database_credential_file_path = "carps/experimenter/credentials.yaml"

experimenter = PyExperimenter(
    experiment_configuration_file_path=experiment_configuration_file_path,
    name="carps",
    database_credential_file_path=database_credential_file_path,
    log_level=logging.INFO,
    use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    use_codecarbon=False
)


column_names = list(experimenter.db_connector.database_configuration.keyfields.keys())
existing_rows = experimenter.db_connector._get_existing_rows(column_names)

# Check if experiments exists
print("Checking if experiments already exist...")
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


BATCH_SIZE = 5000
for i in range(0, len(experiments_to_add), BATCH_SIZE):
    batch = experiments_to_add[i:i+BATCH_SIZE]
    experimenter.fill_table_with_rows(batch)

