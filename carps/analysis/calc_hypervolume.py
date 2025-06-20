"""Calculate hypervolume from trajectory logs."""

from __future__ import annotations

import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from tqdm import tqdm

from carps.analysis.utils import get_ids_mo
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)

run_id = ["task_type", "benchmark_id", "task_id", "optimizer_id", "seed"]


def gather_trajectory(x: pd.DataFrame) -> pd.DataFrame:
    """Gather trajectory data.

    The trajectory is the history of incumbet (best) configurations over one optimization run.

    Args:
        x (pd.DataFrame): Dataframe with the logs.

    Returns:
        pd.DataFrame: Dataframe with the trajectory.
    """
    metadata = dict(zip(run_id, x.name, strict=False))
    data = []
    for n_trials, gdf in x.groupby("n_trials"):
        cost_inc = (
            gdf["trial_value__cost_raw"].apply(eval).apply(lambda x: np.array([np.array(c) for c in x])).to_numpy()
        )
        n_obj = len(cost_inc[0])
        cost_inc = np.concatenate(cost_inc).reshape(-1, n_obj)
        D = {
            "n_trials": n_trials,
            "n_incumbents": len(gdf),
            "trial_value__cost": cost_inc,
            "trial_value__cost_inc": cost_inc,
        }
        D.update(metadata)
        data.append(D)
    return pd.DataFrame(data)


def get_reference_point(x: pd.DataFrame, on_key: str = "trial_value__cost") -> np.ndarray:
    """Get reference point from the dataframe.

    Dataframe should only contain data from one task. The reference point is the maximum
    of the costs over all trials. This is the worst case scenario for the hypervolume
    calculation. The reference point is needed to define the bound of the hypervolume.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_inc".

    Returns:
        np.ndarray: Reference point.
    """
    if "task_id" in x.columns:
        assert x["task_id"].nunique() == 1, "Cannot get reference point for multiple tasks"  # noqa: PD101
    costs = get_costs(x, on_key)
    return np.max(costs, axis=0)


def get_cost_min(x: pd.DataFrame, on_key: str = "trial_value__cost") -> np.ndarray:
    """Get the minimum objective values from the dataframe.

    Dataframe should only contain data from one task. The point is the minimum
    of the costs over all trials. This is the best case scenario for the hypervolume
    calculation. The minimum point is needed for normalization.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_inc".

    Returns:
        np.ndarray: Minimum cost.
    """
    if "task_id" in x.columns:
        assert x["task_id"].nunique() == 1, "Cannot get reference point for multiple tasks"  # noqa: PD101
    costs = get_costs(x, on_key)
    return np.min(costs, axis=0)


def get_costs(x: pd.DataFrame, on_key: str = "trial_value__cost") -> np.ndarray:
    """Get costs from the dataframe.

    Here, it is expected that the costs are vectors (in the case of multi-objective optimization).

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.
        on_key (str, optional): Column to use for the costs. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_raw".
    """
    return np.array(x[on_key].to_list())


def add_reference_point(x: pd.DataFrame, on_key: str = "trial_value__cost") -> pd.DataFrame:
    """Add reference point to the dataframe.

    The reference point is needed to define the bound of the hypervolume.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_inc".

    Returns:
        pd.DataFrame: Dataframe with the reference point.
    """
    reference_point = get_reference_point(x, on_key)
    x["reference_point"] = [reference_point] * len(x)
    minimum_cost = get_cost_min(x, on_key)
    x["minimum_cost"] = [minimum_cost] * len(x)
    return x


def calc_hv(x: pd.DataFrame, on_key: str = "trial_value__cost") -> pd.DataFrame:
    """Calculate hypervolume per trajectory step.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_inc".

    Returns:
        pd.DataFrame: Dataframe with the hypervolume.
    """
    F = get_costs(x, on_key)
    ind = HV(ref_point=x["reference_point"].iloc[0], pf=None, nds=False)
    x["hypervolume"] = ind(F)
    return x


def serialize_array(arr: np.ndarray) -> str:
    """Serialize numpy array to JSON.

    Args:
        arr (np.ndarray): Numpy array.

    Returns:
        str: Serialized numpy array.
    """
    return json.dumps(arr.tolist())


def deserialize_array(serialized_arr: str) -> np.ndarray:
    """Deserialize numpy array from JSON.

    Args:
        serialized_arr (str): Serialized numpy array.

    Returns:
        np.ndarray: Numpy array.
    """
    deserialized = serialized_arr
    try:
        deserialized = np.array(json.loads(serialized_arr))
        print(deserialized)
    except Exception as e:  # noqa: BLE001
        print(e)
        print(serialized_arr)
    return deserialized


def maybe_serialize(x: Any | np.ndarray) -> Any | str:
    """Serialize numpy array to JSON if it is a numpy array.

    Args:
        x (Any | np.ndarray): Input.

    Returns:
        Any | str: Serialized numpy array or input.
    """
    if isinstance(x, np.ndarray):
        return serialize_array(x)
    return x


def maybe_deserialize(x: Any | str) -> Any | np.ndarray:
    """Maybe deserialize numpy array from JSON.

    Args:
        x (Any | str): Input, might be a serialized numpy array.

    Returns:
        Any | np.ndarray: Deserialized numpy array or input.
    """
    if isinstance(x, str):
        return deserialize_array(x)
    return x


def add_hypervolume_to_df(logs: pd.DataFrame, on_key: str = "trial_value__cost") -> pd.DataFrame:
    """Add hypervolume to the dataframe.

    If there are multiple objectives, add reference point and calculate hypervolume.

    Args:
        logs (pd.DataFrame): Dataframe with the logs.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_raw".

    Returns:
        pd.DataFrame: Dataframe with the hypervolume.
    """
    tqdm.pandas(desc="Calc hypervolume...")
    ids_mo = get_ids_mo(logs)
    add_reference_point_partial = partial(add_reference_point, on_key=on_key)
    mo_cols = ["hypervolume", "reference_point"]
    for mo_col in mo_cols:
        if mo_col not in logs.columns:
            logs[mo_col] = None
    if len(ids_mo) > 0:
        logs_mo = logs.loc[ids_mo].groupby(by=["task_id"]).apply(add_reference_point_partial).reset_index(drop=True)
        logs_mo = apply_calc_hv_low_mem(logs_mo, on_key=on_key)
        logs = pd.concat([logs.loc[~ids_mo], logs_mo], axis=0).reset_index(drop=True)
    return logs


def calc_hv_for_run_old(
    group_id: list[str], logs: pd.DataFrame, run_id: list[str], on_key: str = "trial_value__cost"
) -> pd.DataFrame:
    """Calculate hypervolume for a single run.

    Args:
        group_id (list[str]): List of run identifiers.
        logs (pd.DataFrame): Dataframe with the logs.
        run_id (list[str]): List of column names to use for the run identifiers.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_raw".

    Returns:
        pd.DataFrame: Dataframe with the hypervolume for the run.
    """
    gdf = logs[
        (logs[run_id[0]] == group_id[0])
        & (logs[run_id[1]] == group_id[1])
        & (logs[run_id[2]] == group_id[2])
        & (logs[run_id[3]] == group_id[3])
        & (logs[run_id[4]] == group_id[4])
    ]
    # gdf = gdf.compute()
    # gdf2 = logs[logs[run_id].apply(lambda x, group_id=group_id: all(x == group_id), axis=1)].copy()
    # assert len(gdf2) == len(gdf), "log selection failed"
    if len(gdf) == 0:
        return None
    # Sort gdf by n_trials
    gdf = gdf.sort_values(by="n_trials")

    ind = HV(ref_point=gdf["reference_point"].iloc[0], pf=None, nds=False)

    hvs = []
    for n_trial_max in range(len(gdf)):
        F = get_costs(gdf.iloc[: n_trial_max + 1], on_key)
        hv = float(ind(F))
        hvs.append(hv)
    gdf["hypervolume"] = hvs
    # assert that hvs is monotonically increasing
    assert np.all(np.diff(hvs) + 1e-8 >= 0), "Hypervolume is not monotonically increasing"
    return gdf


def calc_hv_for_run(gdf_fn_in: str, on_key: str = "trial_value__cost") -> pd.DataFrame:
    """Calculate hypervolume for a single run.

    Args:
        gdf_fn_in (str): Path to the input dataframe with the logs. Should only contain
            data from one run (one task, one optimizer, one seed).
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_raw".

    Returns:
        pd.DataFrame: Dataframe with the hypervolume for the run.
    """
    gdf_fn_in = Path(gdf_fn_in)  # type: ignore[assignment]
    gdf_fn = Path("tmp/hypervolume") / gdf_fn_in.name  # type: ignore[attr-defined]
    if gdf_fn.is_file():
        return gdf_fn

    gdf = pd.read_parquet(gdf_fn_in, engine="pyarrow")

    gdf_fn.parent.mkdir(parents=True, exist_ok=True)
    if len(gdf) == 0:
        return None
    # Sort gdf by n_trials
    gdf = gdf.sort_values(by="n_trials")

    cost_max = gdf["reference_point"].iloc[0]
    cost_min = gdf["minimum_cost"].iloc[0]
    reference_point = np.ones(len(cost_max))  # We work on normalized objective values

    ind = HV(ref_point=reference_point, pf=None, nds=False)

    hvs = []
    for n_trial_max in range(len(gdf)):
        F = get_costs(gdf.iloc[: n_trial_max + 1], on_key)
        # Normalize
        Fbefore = F
        F = (F - cost_min) / (cost_max - cost_min)
        assert F.shape == Fbefore.shape, f"{F.shape}, {Fbefore.shape}"
        hv = float(ind(F))
        hvs.append(hv)
    gdf["hypervolume"] = hvs
    # assert that hvs is monotonically increasing
    assert np.all(np.diff(hvs) + 1e-7 >= 0), f"Hypervolume is not monotonically increasing, {hvs, np.diff(hvs)}"
    gdf.to_parquet(gdf_fn, index=False, engine="pyarrow")
    return gdf_fn


def apply_calc_hv_low_mem(logs: pd.DataFrame, on_key: str = "trial_value__cost") -> pd.DataFrame:
    """Calculate hypervolume for each run in the logs.

    Args:
        logs (pd.DataFrame): Dataframe with the logs.
        on_key (str, optional): Column to use for the reference point. Defaults to "trial_value__cost".
            Can also be "trial_value__cost_raw".

    Returns:
        pd.DataFrame: Dataframe with the hypervolume for each run.
    """
    logger.info("Calculating hypervolume for each run in the logs...")

    def _save_group_to_filesystem(group: pd.DataFrame, input_directory: Path) -> pd.DataFrame:
        group_id = group[run_id].iloc[0].to_list()
        gdf_fn = input_directory / f"{'_'.join([str(g) for g in group_id]).replace('/','_')}.parquet"
        if gdf_fn.is_file():
            return gdf_fn
        gdf_fn.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(gdf_fn, index=False, engine="pyarrow")
        return gdf_fn

    input_directory = Path("tmp/hypervolume_in")
    output_directory = Path("tmp/hypervolume")
    delete_existing = True
    if delete_existing:
        if input_directory.is_dir() and False:
            for fn in input_directory.iterdir():
                fn.unlink()
        if output_directory.is_dir():
            for fn in output_directory.iterdir():
                fn.unlink()
    input_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    tqdm.pandas(desc="Saving groups to filesystem...")
    gdf_fns = logs.groupby(by=run_id).progress_apply(
        lambda x, input_directory=input_directory: _save_group_to_filesystem(x, input_directory)
    )
    gdf_fns = os.listdir(input_directory)
    gdf_fns = [Path(input_directory) / fn for fn in gdf_fns]
    del logs

    partial_func = partial(calc_hv_for_run, on_key=on_key)

    logger.info(
        f"...processing this can take a while. Check progress with `ls {output_directory!s} | wc -l` "
        f"({len(gdf_fns)} tasks in total)."
    )
    with Pool() as pool:
        results = pool.map(partial_func, gdf_fns)

    results = [pd.read_parquet(r, engine="pyarrow") for r in results if r is not None]
    return pd.concat(results, ignore_index=True).reset_index(drop=True)


def load_trajectory(rundir: str) -> pd.DataFrame:
    """Load trajectory data from rundir.

    Assumes the data lies in Path(rundir) / "trajectory.parquet".

    Args:
        rundir (str): Directory with the trajectory data.

    Returns:
        pd.DataFrame: Dataframe with the trajectory data.
    """
    fn = Path(rundir) / "trajectory.parquet"
    if not fn.is_file():
        raise ValueError(f"Cannot find {fn}. Did you run `python -m carps.analysis.calc_hypervolume {rundir}`?")
    df = pd.read_parquet(fn)  # noqa: PD901
    df = df.map(maybe_deserialize)  # noqa: PD901
    print(df["trial_value__cost"].iloc[0], type(df["trial_value__cost"].iloc[0]))
