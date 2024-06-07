from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def norm_by_opt(df: pd.DataFrame, optimizer_id: str) -> pd.DataFrame:
    df_new = []
    for _gid, gdf in df.groupby(by=["problem_id", "seed"]):
        reference = gdf[gdf["optimizer_id"] == optimizer_id]["trial_value__cost"]
        gdf["trial_value__cost_normopt"] = gdf.groupby("optimizer_id")["trial_value__cost"].transform(
            lambda x: x / reference
        )
        df_new.append(gdf)
    df = pd.concat(df_new).reset_index(drop=True)

    # df["trial_value_cost_normopt"] = df.groupby("problem_id").apply(_norm_by_opt)
    df["trial_value__cost_inc_normopt"] = df.groupby(by=["problem_id", "optimizer_id", "seed"])[
        "trial_value__cost_normopt"
    ].transform("cummin")
    return df


if __name__ == "__main__":
    df = pd.read_parquet(
        "/scratch/hpc-prf-intexml/cbenjamins/repos/CARP-S-Experiments/lib/CARP-S/runs/RandomSearch/MFPBench/logs.parquet"
    )

    df_new = df.copy()
    df_new["optimizer_id"] = "HeheOpt"
    df_new["trial_value__cost"] -= 0.1
    df = pd.concat([df, df_new]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    normalize_by_opt = "RandomSearch"
    df = norm_by_opt(df, "RandomSearch")

    sns.lineplot()
