from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger("Subselect")


def parse_metadata(metadata: str) -> dict:
    """Parse metadata string into a dictionary.

    Args:
        metadata (str): Metadata string to parse.
            Can look like this "n=15,k=12,dim=3, discrepancy=0.442226, runtime=0.000980".

    Returns:
        dict: Parsed metadata as a dictionary.
    """
    return  {key.strip(" "): float(value) if "." in value else int(value) 
               for key, value in (item.split("=") for item in metadata.split(","))}

def subselect(points_df: pd.DataFrame, k: int, n_reps: int = 5000, executable: str = "./a.out") -> tuple[pd.DataFrame, dict]:
    """Subselect k points from a given dataframe using an external executable.

    Args:
        points_df (pd.DataFrame): DataFrame containing the points to be subselected. Index can be the task id and the
            columns can be different optimizers.
        k (int): Number of points to select.
        n_reps (int, optional): Number of repetitions for the selection process. Defaults to 5000.
        executable (str, optional): Path to the external executable. Defaults to "./a.out".

    Returns:
        tuple[pd.DataFrame, dict]: DataFrame containing the selected points and metadata.
    """
    logger.info(f"Subselecting {k} points from {len(points_df)} points.")
    pointfile = f"tmpfile_in_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt"
    outfile = f"tmpfile_out_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt"

    points_df.to_csv(pointfile, sep=" ", index=False, header=False)
    n_points, dimension = points_df.to_numpy().shape

    command = f"export SHIFT_TRIES={n_reps}; {executable} {pointfile} {dimension} {n_points} {k} {outfile}"
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
    has_errored = result.returncode != 0
    if has_errored:
        print(result.stderr)
        raise ValueError(f"Subselection failed with error code {result.returncode}")

    with open(outfile) as f:
        lines = f.readlines()
        metadata_str = lines[0]
        lines = "".join(lines[1:])
    with open(outfile, "w") as f:
        f.write(lines)

    metadata = parse_metadata(metadata_str)
    metadata["n_reps"] = n_reps
    subset_df = pd.read_csv(outfile, sep=" ", header=None)
    subset_df.columns = points_df.columns

    # Match task ids
    subset_points = subset_df.to_numpy()
    fullset_points = points_df.to_numpy()
    index = [points_df.index[np.all(np.isclose(fullset_points, row), axis=1)][0] for row in subset_points]
    subset_df.index = index
    subset_df.index.name = points_df.index.name
    metadata["task_ids"] = index

    if Path(outfile).exists():
        Path(outfile).unlink()
    Path(pointfile).unlink()

    return subset_df, metadata

def select_sets_split(
        points_df: pd.DataFrame, k: int, n_reps: int = 5000, executable: str = "./a.out",
        subset_ids: tuple[str] = ("dev", "test")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subselect sets of points from a given DataFrame.

    We first select a subset of 2*k from the points_df. Afterwards, we select k points from this subset
    to split the subset into two.

    Args:
        points_df (pd.DataFrame): DataFrame containing the points.
        k (int): Number of points to select in each subset.
        n_reps (int): Number of repetitions for the selection.
        executable (str): Path to the executable for subsetting.
        subset_ids (tuple[str]): Identifiers for the subsets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the selected subsets and their metadata.
    """
    logger.info("Starting subselecting sets by splitting...")
    n_sets = len(subset_ids)
    assert n_sets == 2, "Subset IDs must be a tuple of two elements."  # noqa: PLR2004
    subsets = []
    metadatas = []
    fullset_df = points_df.copy()
    logger.info("...select reduced set")
    reduced_fullset_df, metadata = subselect(fullset_df, n_sets*k, n_reps, executable)

    for subset_id in subset_ids:
        logger.info(f"...select subset {subset_id}")
        subset_df, metadata = subselect(reduced_fullset_df, k, n_reps, executable)
        subset_df["subset_id"] = subset_id
        subsets.append(subset_df)
        metadata["subset_id"] = subset_id
        metadatas.append(pd.Series(metadata))
        reduced_fullset_df = reduced_fullset_df[~reduced_fullset_df.index.isin(subset_df.index)]
    subset_df = pd.concat(subsets)
    metadata = pd.concat(metadatas)
    return subset_df, metadata

def select_sets_sequential(
        points_df: pd.DataFrame, k: int, n_reps: int = 5000, executable: str = "./a.out",
        subset_ids: tuple[str] = ("dev", "test")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subselect sets of points from a given DataFrame.

    We select len(subset_ids) subsets of size k from the points_df. After each subset is selected,
    the points are removed from the full set to avoid overlap.
    The subsets are identified by the subset_ids.

    Args:
        points_df (pd.DataFrame): DataFrame containing the points.
        k (int): Number of points to select in each subset.
        n_reps (int): Number of repetitions for the selection.
        executable (str): Path to the executable for subsetting.
        subset_ids (tuple[str]): Identifiers for the subsets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the selected subsets and their metadata.
    """
    logger.info("Starting subselecting sets sequentially...")
    subsets = []
    metadatas = []
    fullset_df = points_df.copy()
    for subset_id in subset_ids:
        logger.info(f"...select subset {subset_id}")
        subset_df, metadata = subselect(fullset_df, k, n_reps, executable)
        subset_df["subset_id"] = subset_id
        subsets.append(subset_df)
        metadata["subset_id"] = subset_id
        metadatas.append(pd.Series(metadata))
        fullset_df = fullset_df[~fullset_df.index.isin(subset_df.index)]
    subset_df = pd.concat(subsets)
    metadata = pd.concat(metadatas)
    return subset_df, metadata

if __name__ == "__main__":
    fire.Fire(select_sets_sequential)