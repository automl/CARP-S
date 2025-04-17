"""Process logs from database.

Prerequisites:
    The database must be downloaded with `python -m carps.experimenter.database.download_results`.
    The logs will be in the `experimenter/results` directory.
"""

from __future__ import annotations

from functools import partial
from multiprocessing import Pool  # noqa: F401
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from carps.analysis.gather_data import maybe_postadd_task, process_logs
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


def filter_non_incumbent_entries(logs: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-incumbent entries from the logs.

    This can be useful if metadata needs to be matched. The dataframes
    can be reduced by ~90%.

    Args:
        logs (pd.DataFrame): The logs DataFrame containing trial information.

    Returns:
        pd.DataFrame: A DataFrame containing only the incumbent entries.
    """
    return logs[logs["trial_value__cost_inc"] == logs["trial_value__cost"]]


def add_metadata(
    logs_from_one_run: pd.DataFrame, experiment_id: int, experiment_config_table: pd.DataFrame
) -> pd.DataFrame:
    """Add metadata to the logs from a single run.

    Args:
        logs_from_one_run (pd.DataFrame): The logs DataFrame containing trial information.
        experiment_id (int): The ID of the experiment.
        experiment_config_table (pd.DataFrame): The experiment configuration table.

    Returns:
        pd.DataFrame: A DataFrame containing the logs with added metadata.
    """
    ignore_columns = [
        "creation_date",
        "start_date",
        "end_date",
        "error",
        "machine",
        "slurm_job_id",
        "status",
        "config",
        "config_hash",
        "name",
    ]
    metadata_columns = [c for c in experiment_config_table.columns if c not in ignore_columns]

    metadata_row = experiment_config_table[experiment_config_table["ID"] == experiment_id].iloc[0][metadata_columns]
    metadata_dict = metadata_row.to_dict()
    if "ID" in metadata_dict:
        metadata_dict["experiment_config_id"] = metadata_dict.pop("ID")

    logs_from_one_run = logs_from_one_run.copy()
    for k, v in metadata_dict.items():
        logs_from_one_run.loc[:, k] = v

    return maybe_postadd_task(logs_from_one_run)


def process_single_run_from_database(
    logs_from_one_run: pd.DataFrame,
    experiment_config_table: pd.DataFrame,
    only_incumbents: bool = True,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """Process logs from a single run.

    Args:
        logs_from_one_run (pd.DataFrame): The logs DataFrame containing trial information.
        experiment_config_table (pd.DataFrame): The experiment configuration table.
        only_incumbents (bool, default True): Whether to filter out non-incumbent entries. This speeds up adding
            metadata significantly.

    Raises:
        ValueError: If multiple values for `experiment_id` are found in the logs.

    Returns:
        pd.DataFrame: A DataFrame containing the processed logs.
    """
    if logs_from_one_run["experiment_id"].nunique() != 1:  # noqa: PD101
        raise ValueError("Multiple values for `experiment_id` found in the logs. Something is suspicious.")
    experiment_id = logs_from_one_run["experiment_id"].iloc[0]
    logs_from_one_run = process_logs(logs_from_one_run)
    if only_incumbents:
        logs_from_one_run = filter_non_incumbent_entries(logs=logs_from_one_run)
    return add_metadata(
        logs_from_one_run=logs_from_one_run,
        experiment_id=experiment_id,
        experiment_config_table=experiment_config_table,
    )


def process_experiment(
    experiment_id: int,
    logs_from_database: pd.DataFrame,
    experiment_config_table: pd.DataFrame,
    only_incumbents: bool = True,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """Process logs for a specific experiment.

    Args:
        experiment_id (int): The ID of the experiment.
        logs_from_database (pd.DataFrame): The logs DataFrame containing trial information.
        experiment_config_table (pd.DataFrame): The experiment configuration table.
        only_incumbents (bool, default True): Whether to filter out non-incumbent entries. This speeds up adding
            metadata significantly.

    Returns:
        pd.DataFrame: A DataFrame containing the processed logs for the specific experiment.
    """
    # Filter the logs for the current experiment
    logs_for_experiment = logs_from_database[logs_from_database["experiment_id"] == experiment_id].copy()

    # Call the function to process logs for the specific experiment
    return process_single_run_from_database(
        logs_from_one_run=logs_for_experiment,
        experiment_config_table=experiment_config_table,
        only_incumbents=only_incumbents,
    )


def process_logs_from_database(
    logs_from_database_filename: str = "trials.parquet",
    experiment_config_table_filename: str = "experiment_config.parquet",
    output_filename: str = "processed_logs.parquet",
    results_dir: str = "experimenter/results",
    only_incumbents: bool = True,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """Process logs from the database with multiprocessing for speed-up.

    Args:
        logs_from_database_filename (str): The path to the logs DataFrame containing trial information.
        experiment_config_table_filename (str): The path to the experiment configuration table.
        output_filename (str): The path to save the processed logs.
        results_dir (str, default "experimenter/results"): The directory where the results are saved.
        only_incumbents (bool, default True): Whether to filter out non-incumbent entries. This speeds up adding
            metadata significantly.

    Returns:
        pd.DataFrame: A DataFrame containing the processed logs.
    """
    logger.info("Processing logs from the database...")
    logger.info(f"Results directory: {results_dir}")
    logger.info("(Did you download the data with `python -m carps.experimenter.database.download_results`?)")
    results_dir = Path(results_dir)  # type:ignore[assignment]
    output_filename = results_dir / output_filename  # type:ignore[operator]
    logs_from_database = pd.read_parquet(results_dir / logs_from_database_filename)  # type:ignore[operator]
    experiment_config_table = pd.read_parquet(results_dir / experiment_config_table_filename)  # type:ignore[operator]

    # Get unique experiment ids
    experiment_ids = logs_from_database["experiment_id"].unique()

    # Prepare a partial function for process_experiment with pre-filled arguments
    process_experiment_partial = partial(
        process_experiment,
        logs_from_database=logs_from_database,
        experiment_config_table=experiment_config_table,
        only_incumbents=only_incumbents,
    )

    # Set up multiprocessing pool to process the logs
    # with Pool() as pool:
    #     # Wrap pool.imap_unordered with tqdm to show the progress bar
    #     result = list(tqdm(
    #           pool.imap_unordered(
    #           process_experiment_partial, experiment_ids), total=len(experiment_ids), desc="Processing experiments"))
    logger.info(f"Start processing {len(experiment_ids)} experiments... This might take a while...")
    result = [
        process_experiment_partial(experiment_id)
        for experiment_id in tqdm(experiment_ids, desc="Processing experiments")
    ]

    # Combine the results into a single DataFrame
    processed_logs = pd.concat(result, ignore_index=True).reset_index(drop=True)
    processed_logs.to_parquet(output_filename, index=False)
    logger.info(f"Processed logs saved to {output_filename} 💌.")
    return processed_logs


if __name__ == "__main__":
    fire.Fire(process_logs_from_database)
