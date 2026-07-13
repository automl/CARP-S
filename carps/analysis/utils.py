"""Utility functions for analysis."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D

from carps.utils.loggingutils import get_logger

# colorblind_palette = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"] # noqa: E501
colorblind_palette = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#7A7A7A"]
logger = get_logger("analysis utils")
markers = list(Line2D.filled_markers)


def get_marker_palette(
    df: pd.DataFrame | None = None, model_name_key: str = "optimizer_id", optimizers: list[str] | None = None
) -> dict[str, Any]:
    """Get a marker palette based on the optimizers.

    Args:
        df (pd.DataFrame, optional): Results dataframe.
        model_name_key (str, optional): The column name for the model name. Defaults to "model_name".
        optimizers (list[str], optional): List of optimizers. If None, will be extracted from df. Defaults to None.

    Returns:
        dict[str, Any]: Marker map.
    """
    if optimizers is None:
        assert df is not None, "Either df or optimizers must be provided."
        optimizers = list(df[model_name_key].unique())
    optimizers.sort()
    if len(optimizers) > len(markers):
        logger.info(f"Too many optimizers: {len(optimizers)} > {len(markers)}. Reusing markers.")
    return dict(zip(optimizers, itertools.cycle(markers), strict=False))


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
    if len(optimizers) > len(colormaps):
        logger.info(f"Too many optimizers: {len(optimizers)} > {len(colormaps)}. Using continuous colormap.")
        n_optimizers = len(optimizers)
        colormaps = plt.colormaps.get_cmap("viridis")(np.linspace(0, 1, n_optimizers))
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


def filter_only_final_performance(
    df: pd.DataFrame | pl.DataFrame,
    x_column: str = "n_trials_norm",
    max_x: float = 1,
    key_performance: str = "trial_value__cost_inc",
) -> pd.DataFrame | pl.DataFrame:
    """Extracts the best-found performance (incumbent) for each experimental run
    within a specified budget constraint.

    This function simulates a snapshot of an optimization process. It first
    constrains the data to a maximum budget (x_column) and then identifies
    the single best configuration found up to that point for every unique
    combination of optimizer, task, and random seed.

    Algorithm Logic:
    1. Filter: Retain only observations where the budget metric is <= `max_x`.
    2. Group: Partition data by ["optimizer_id", "task_id", "seed"].
    3. Identify Incumbent: Within each partition, locate the observation
       with the minimum value in `key_performance`.
    4. Tie-breaking: If multiple timestamps share the same minimum cost,
       the earliest occurrence is retained.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        The dataset containing optimization traces. Supports both Pandas and
        Polars backends.
    x_column : str, optional
        The budget or time-step column (e.g., normalized trials, wall-clock
        time, or iterations), by default "n_trials_norm".
    max_x : float, optional
        The budget cutoff. Any data points beyond this value are ignored
        to simulate early stopping or specific budget analysis, by default 1.
    key_performance : str, optional
        The metric to be minimized (e.g., cost, regret, or error).
        By default "trial_value__cost_inc".

    Returns:
    -------
    Union[pd.DataFrame, pl.DataFrame]
        A reduced DataFrame containing exactly one row per (optimizer, task, seed),
        representing the peak performance achieved within the given budget.
        The return type matches the input type.

    Raises:
    ------
    TypeError
        If the input 'df' is neither a Pandas nor a Polars DataFrame.
    """
    group_cols = ["optimizer_id", "task_id", "seed"]

    # --- Polars Backend (Vectorized Expressions) ---
    if isinstance(df, pl.DataFrame):
        return (
            df.filter(pl.col(x_column) <= max_x)
            .sort(key_performance, descending=False)
            .group_by(group_cols, maintain_order=True)
            .first()
        )

    # --- Pandas Backend (Vectorized Sorting/Grouping) ---
    if isinstance(df, pd.DataFrame):
        # We avoid .apply() as it is slow; sorting + .first() is the idiomatic alternative
        return (
            df[df[x_column] <= max_x]
            .sort_values(key_performance, ascending=True)
            .groupby(group_cols, as_index=False)
            .first()
        )

    raise TypeError(f"Unsupported dataframe type: {type(df)}. Expected Pandas or Polars.")


def convert_mixed_types_to_str(logs: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Convert mixed type columns to str.

    Necessary to be able to write a parquet file.

    Args:
        logs (pd.DataFrame): Logs.
        logger (logging.Logger, optional): Logger. Defaults to None.

    Returns:
        pd.DataFrame: Logs with mixed type columns converted
    """
    mixed_type_columns = logs.select_dtypes(include=["O", "object"]).columns
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


def determine_filename_id(group_keys: Sequence[str], gid: list[Any]) -> str:
    """Determine filename id based on group keys.

    Parameters
    ----------
    group_keys : Sequence[str]
        The group keys.
    gid : list[Any]
        The group values.

    Returns:
    -------
    str
        The filename id.
    """
    return "_".join([f"{k}-{v}" for k, v in zip(group_keys, gid, strict=True)])


def get_figure_title(group_keys: Sequence[str], gid: list[Any]) -> str:
    """Determine filename id based on group keys.

    Parameters
    ----------
    group_keys : Sequence[str]
        The group keys.
    gid : list[Any]
        The group values.

    Returns:
    -------
    str
        The filename id.
    """
    return ",".join([f"{k}: {v}" for k, v in zip(group_keys, gid, strict=True)])
