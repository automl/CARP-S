"""Plot relative performance of optimizers."""

from __future__ import annotations

import pandas as pd


def norm_by_opt(df: pd.DataFrame, optimizer_id: str) -> pd.DataFrame:
    """Normalize the cost by the cost of a reference optimizer.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the logs.

    optimizer_id : str
        The optimizer to normalize by.

    Returns:
    -------
    pd.DataFrame
        Dataframe with normalized costs.
    """
    df_new = []
    for _gid, gdf in df.groupby(by=["task_id", "seed"]):
        reference = gdf[gdf["optimizer_id"] == optimizer_id]["trial_value__cost"]

        gdf["trial_value__cost_normopt"] = gdf.groupby("optimizer_id")["trial_value__cost"].transform(
            lambda x, reference=reference: x / reference
        )
        df_new.append(gdf)
    df = pd.concat(df_new).reset_index(drop=True)  # noqa: PD901

    # df["trial_value_cost_normopt"] = df.groupby("task_id").apply(_norm_by_opt)
    df["trial_value__cost_inc_normopt"] = df.groupby(by=["task_id", "optimizer_id", "seed"])[
        "trial_value__cost_normopt"
    ].transform("cummin")
    return df
