from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import fire
from carps.analysis.gather_data import convert_mixed_types_to_str
from pymoo.indicators.hv import HV


run_id = ["scenario", "set", "benchmark_id", "problem_id", "optimizer_id", "seed"]

def gather_trajectory(x: pd.DataFrame) -> pd.DataFrame:
    metadata = dict(zip(run_id, x.name))
    data = []
    for n_trials, gdf in x.groupby("n_trials"):
        cost_inc = gdf["trial_value__cost_raw"].apply(eval).apply(lambda x: np.array([np.array(c) for c in x])).to_numpy()
        n_obj =  len(cost_inc[0])
        cost_inc = np.concatenate(cost_inc).reshape(-1,n_obj)
        D = {
            "n_trials": n_trials,
            "n_incumbents": len(gdf),
            "trial_value__cost_inc": cost_inc,
        }
        D.update(metadata)
        data.append(D)
    return pd.DataFrame(data)

def add_reference_point(x: pd.DataFrame) -> pd.DataFrame:
    costs = x["trial_value__cost_inc"].apply(lambda x: np.array([np.array(c) for c in x])).to_list()
    costs = np.concatenate(costs)
    reference_point = np.max(costs, axis=0)
    x["reference_point"] = [reference_point] * len(x)
    return x

def calc_hv(x: pd.DataFrame) -> pd.DataFrame:
    F = np.concatenate(np.array([np.array(p) for p in x["trial_value__cost_inc"].to_numpy()]))

    ind = HV(ref_point=x["reference_point"].iloc[0], pf=None, nds=False)
    x["hypervolume"] = ind(F)
    return x

def calculate_hypervolume(rundir: str) -> None:
    fn = Path(rundir) / "logs.parquet"
    if not fn.is_file():
        raise ValueError(f"Cannot find {fn}. Did you run `python -m carps.analysis.gather_data {rundir} trajectory_logs.jsonl`?")
    df = pd.read_parquet(fn)
    if df["scenario"].nunique() > 2 or df["scenario"].unique()[0] != "multi-objective":
        raise ValueError(f"Oops, found some non multi-objective logs in {fn}. This might not work...")
    trajectory_df = df.groupby(by=run_id).apply(gather_trajectory).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=["scenario", "set", "problem_id"]).apply(add_reference_point).reset_index(drop=True)
    trajectory_df = trajectory_df.groupby(by=run_id + ["n_trials"]).apply(calc_hv).reset_index(drop=True)
    trajectory_df.to_csv(Path(rundir) / "trajectory.csv")
    trajectory_df = convert_mixed_types_to_str(trajectory_df)
    trajectory_df.to_parquet(Path(rundir) / "trajectory.parquet")

if __name__ == "__main__":
    fire.Fire(calculate_hypervolume)