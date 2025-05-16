"""Calculate hypervolume from trajectory logs."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from tqdm import tqdm

from carps.analysis.utils import convert_mixed_types_to_str

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


def calculate_hypervolume(rundir: str) -> None:
    """Calculate hypervolume from trajectory logs.

    Save to rundir / "trajectory.parquet" and rundir / "trajectory.csv".

    Args:
        rundir (str): Directory with the logs.
    """
    fn = Path(rundir) / "logs.parquet"
    if not fn.is_file():
        raise ValueError(
            f"Cannot find {fn}. Did you run `python -m carps.analysis.gather_data {rundir} trajectory_logs.jsonl`?"
        )
    df = pd.read_parquet(fn)  # noqa: PD901
    if df["task_type"].nunique() > 2 or df["task_type"].unique()[0] != "multi-objective":  # noqa: PLR2004
        raise ValueError(f"Oops, found some non multi-objective logs in {fn}. This might not work...")
    trajectory_df = df.groupby(by=run_id).apply(gather_trajectory).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=["task_type", "task_id"]).apply(add_reference_point).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=[*run_id, "n_trials"]).apply(calc_hv).reset_index(drop=True)
    trajectory_df.to_csv(Path(rundir) / "trajectory.csv")
    trajectory_df = convert_mixed_types_to_str(trajectory_df)
    trajectory_df.to_parquet(Path(rundir) / "trajectory.parquet")


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
    tqdm.pandas(desc="Calc hypervolumne...")
    ids_mo = logs["task_type"] == "multi-objective"
    add_reference_point_partial = partial(add_reference_point, on_key=on_key)
    calc_hv_partial = partial(calc_hv, on_key=on_key)
    mo_cols = ["hypervolume", "reference_point"]
    for mo_col in mo_cols:
        if mo_col not in logs.columns:
            logs[mo_col] = None
    if len(ids_mo) > 0:
        logs.loc[ids_mo] = (
            logs.loc[ids_mo]
            .groupby(by=["task_type", "task_id"])
            .apply(add_reference_point_partial)
            .reset_index(drop=True)
        )
        logs.loc[ids_mo] = (
            logs.loc[ids_mo].groupby(by=[*run_id, "n_trials"]).progress_apply(calc_hv_partial).reset_index(drop=True)
        )
    return logs


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


if __name__ == "__main__":
    fire.Fire(calculate_hypervolume)
