"""Utility functions for analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


colorblind_palette = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"]


def get_color_palette(
    df: pd.DataFrame | None = None, model_name_key: str = "optimizer_id", optimizers: list[str] | None = None
) -> dict[str, Any]:
    """Get a color palette based on the optimizers.

    Args:
        df (pd.DataFrame, optional): Results dataframe.
        model_name_key (str, optional): The column name for the model name. Defaults to "model_name".
        optimizers (list[str], optional): List of optimizers. If None, will be extracted from df. Defaults to None.

    Returns:
        dict[str, Any]: Color map.
    """
    if optimizers is None:
        assert df is not None, "Either df or optimizers must be provided."
        optimizers = list(df[model_name_key].unique())
    optimizers.sort()
    cmap1 = colorblind_palette
    cmap2 = sns.color_palette("colorblind", as_cmap=False)
    cmap3 = sns.color_palette("Paired", as_cmap=False)
    colormaps = list(cmap1) + list(cmap2) + list(cmap3)
    assert len(optimizers) <= len(colormaps), f"Too many optimizers: {len(optimizers)} > {len(colormaps)}"
    return dict(zip(optimizers, colormaps, strict=False))


def savefig(fig: plt.Figure, filename: str | Path) -> None:
    """Save figure as png and pdf.

    Args:
        fig (plt.Figure): Figure to save.
        filename (str | Path): Filename without extension.
    """
    figure_filename = Path(filename)
    figure_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_filename) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(figure_filename) + ".pdf", dpi=300, bbox_inches="tight")


def setup_seaborn(font_scale: float | None = None) -> None:
    """Setup seaborn for plotting.

    Use whitegrid and colorblind palette by default.

    Args:
        font_scale (float | None, optional): Font scale. Defaults to None.
    """
    if font_scale is not None:
        sns.set_theme(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")


def filter_only_final_performance(df: pd.DataFrame, x_column: str = "n_trials_norm", max_x: float = 1) -> pd.DataFrame:
    """Filter final performance based on the maximum x value.

    (1) Filter s.t. the x_column is less than or equal to max_x.
    (2) For each run (each group of optimizer_id, task_id, and seed), keep only the row with the
    best solution, which is defined as the row with the minimum cost_inc value.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the performance data.
    x_column : str, optional
        The column to filter on, by default "n_trials_norm".
    max_x : float, optional
        The maximum value of the x_column to filter by, by default 1.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing only the final performance data for each optimizer, task, and seed.
    """

    def keep(groupdf: pd.DataFrame) -> pd.DataFrame:
        groupdf = groupdf[groupdf[x_column] <= max_x]
        return groupdf[groupdf["trial_value__cost_inc"] == groupdf["trial_value__cost_inc"].min()].iloc[[-1]]

    df_final = df.groupby(["optimizer_id", "task_id", "seed"]).apply(keep, include_groups=False)

    if "level_3" in df_final.columns:
        df_final = df_final.drop(columns=["level_3"])
    return df_final.reset_index()


def convert_mixed_types_to_str(logs: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Convert mixed type columns to str.

    Necessary to be able to write a parquet file.

    Args:
        logs (pd.DataFrame): Logs.
        logger (logging.Logger, optional): Logger. Defaults to None.

    Returns:
        pd.DataFrame: Logs with mixed type columns converted
    """
    mixed_type_columns = logs.select_dtypes(include=["O"]).columns
    if logger:
        logger.debug(f"Goodbye all mixed data, ruthlessly converting {mixed_type_columns} to str...")
    for c in mixed_type_columns:
        # D = logs[c]
        # logs.drop(columns=c)
        if c == "cfg_str":
            continue
        logs[c] = logs[c].map(lambda x: str(x))
        logs[c] = logs[c].astype("str")
    return logs


def percent_budget_used(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the percentage of budget used for each optimizer, task, and seed.

    This function groups the DataFrame by run (optimizer_id, task_id, and seed),
    and calculates the percentage of budget used based on the maximum number of trials.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the performance data.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the percentage of budget used for each optimizer, task, and seed.
    """

    def keep(groupdf: pd.DataFrame) -> pd.DataFrame:
        total_budget = groupdf["task.optimization_resources.n_trials"].max()
        groupdf = groupdf[groupdf["n_trials"] == groupdf["n_trials"].max()].copy()
        groupdf.loc[:, "percent_budget_used"] = groupdf["n_trials"] / total_budget
        return groupdf

    return df.groupby(by=["optimizer_id", "task_id", "seed"]).apply(keep, include_groups=False)


def get_ids_mo(logs: pd.DataFrame) -> pd.Series:
    """Get multi-objective ids.

    Args:
        logs (pd.DataFrame): Logs.

    Returns:
        pd.Series: Boolean series indicating multi-objective ids.
    """
    # TODO determine MO ids by type of cost (first apply maybe_convert_cost_dtype)
    return logs["task_type"].isin(["multi-objective", "multi-fidelity-objective"])
