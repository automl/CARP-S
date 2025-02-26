"""Performance over time analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from carps.analysis.utils import filter_only_final_performance, get_color_palette, savefig, setup_seaborn

if TYPE_CHECKING:
    import pandas as pd


def get_order_by_mean(df: pd.DataFrame, budget_var: str = "n_trials_norm") -> list[str]:
    """Get order of optimizers by mean performance.

    Args:
        df (pd.DataFrame): Results dataframe.
        budget_var (str, optional): Budget variable. Defaults to "n_trials_norm". Necessary to get the final performance
            of the optimizers.

    Returns:
        list[str]: Optimizer order.
    """
    final_df = filter_only_final_performance(df, budget_var=budget_var)
    reduced = final_df.groupby(by="optimizer_id")["trial_value__cost_inc_norm"].apply(np.nanmean)
    reduced = reduced.sort_values()
    return reduced.index.tolist()


def plot_performance_over_time(
    df: pd.DataFrame,
    x: str = "n_trials_norm",
    y: str = "cost_inc_norm",
    hue: str = "optimizer_id",
    figure_filename: str = "figures/performance_over_time",
    figsize: tuple[int, int] = (6, 4),
    show_legend: bool = True,  # noqa: FBT001, FBT002
    title: str | None = None,
    **lineplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot performance over trials/time as a lineplot.

    Args:
        df (pd.DataFrame): Dataframe with the logs.
        x (str, optional): x-axis column. Defaults to "n_trials_norm".
        y (str, optional): y-axis column. Defaults to "cost_inc_norm".
        hue (str, optional): Hue column. Defaults to "optimizer_id".
        figure_filename (str, optional): Figure filename. Defaults to "figures/performance_over_time".
        figsize (tuple[int, int], optional): Figure size. Defaults to (6, 4).
        show_legend (bool, optional): Show legend. Defaults to True.
        title (str | None, optional): Title. Defaults to None.
        **lineplot_kwargs: Additional lineplot arguments.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]: Figure and axes
    """
    setup_seaborn(font_scale=1.5)
    sorter = get_order_by_mean(df=df, budget_var=x)
    df = df.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))  # noqa: PD901
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, **lineplot_kwargs, ax=ax)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.get_legend().remove()
    if title is not None:
        ax.set_title(title)
    savefig(fig, figure_filename)
    return fig, ax


def plot_rank_over_time(
    df: pd.DataFrame,
    x: str = "n_trials_norm",
    y: str = "cost_inc_norm",
    hue: str = "optimizer_id",
    figure_filename: str = "figures/performance_over_time",
    figsize: tuple[int, int] = (6, 4),
    show_legend: bool = True,  # noqa: FBT001, FBT002
    **lineplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot ranks over trials/time as a lineplot.

    Args:
        df (pd.DataFrame): Dataframe with the logs.
        x (str, optional): x-axis column. Defaults to "n_trials_norm".
        y (str, optional): y-axis column. Defaults to "cost_inc_norm".
        hue (str, optional): Hue column. Defaults to "optimizer_id".
        figure_filename (str, optional): Figure filename. Defaults to "figures/performance_over_time".
        figsize (tuple[int, int], optional): Figure size. Defaults to (6, 4).
        show_legend (bool, optional): Show legend. Defaults to True.
        **lineplot_kwargs: Additional lineplot arguments.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]: Figure and axes
    """
    setup_seaborn(font_scale=1.5)
    sorter = get_order_by_mean(df=df)
    df = df.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))  # noqa: PD901
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, **lineplot_kwargs, ax=ax)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.get_legend().remove()
    savefig(fig, figure_filename)
    return fig, ax
