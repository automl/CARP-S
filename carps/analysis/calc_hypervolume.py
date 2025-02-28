"""Calculate hypervolume from trajectory logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from carps.analysis.gather_data import convert_mixed_types_to_str

run_id = ["scenario", "benchmark_id", "task_id", "optimizer_id", "seed"]


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


def add_reference_point(x: pd.DataFrame) -> pd.DataFrame:
    """Add reference point to the dataframe.

    The reference point is needed to define the bound of the hypervolume.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.

    Returns:
        pd.DataFrame: Dataframe with the reference point.
    """
    costs = x["trial_value__cost_inc"].apply(lambda x: np.array([np.array(c) for c in x])).to_list()
    costs = np.concatenate(costs)
    reference_point = np.max(costs, axis=0)
    x["reference_point"] = [reference_point] * len(x)
    return x


def calc_hv(x: pd.DataFrame) -> pd.DataFrame:
    """Calculate hypervolume per trajectory step.

    Args:
        x (pd.DataFrame): Dataframe with the trajectory.

    Returns:
        pd.DataFrame: Dataframe with the hypervolume.
    """
    F = np.concatenate(np.array([np.array(p) for p in x["trial_value__cost_inc"].to_numpy()]))

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
    if df["scenario"].nunique() > 2 or df["scenario"].unique()[0] != "multi-objective":  # noqa: PLR2004
        raise ValueError(f"Oops, found some non multi-objective logs in {fn}. This might not work...")
    trajectory_df = df.groupby(by=run_id).apply(gather_trajectory).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=["scenario", "task_id"]).apply(add_reference_point).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=[*run_id, "n_trials"]).apply(calc_hv).reset_index(drop=True)
    trajectory_df.to_csv(Path(rundir) / "trajectory.csv")
    trajectory_df = convert_mixed_types_to_str(trajectory_df)
    trajectory_df.to_parquet(Path(rundir) / "trajectory.parquet")


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
