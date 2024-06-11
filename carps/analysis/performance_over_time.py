from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from carps.analysis.utils import get_color_palette, savefig, setup_seaborn
from carps.analysis.utils import filter_only_final_performance


if TYPE_CHECKING:
    import pandas as pd

def get_order_by_mean(df: pd.DataFrame) -> list[str]:
    final_df = filter_only_final_performance(df)
    reduced = final_df.groupby(by="optimizer_id")["trial_value__cost_inc_norm"].apply(np.nanmean)
    reduced = reduced.sort_values()
    return reduced.index.tolist()


def plot_performance_over_time(
    df: pd.DataFrame,
    x="n_trials_norm",
    y="cost_inc_norm",
    hue="optimizer_id",
    figure_filename: str = "figures/performance_over_time.pdf",
    figsize: tuple[int, int] = (6, 4),
    show_legend: bool = True,
    title: str | None = None,
    **lineplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    setup_seaborn(font_scale=1.5)
    sorter = get_order_by_mean(df=df)
    df = df.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))
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
    x="n_trials_norm",
    y="cost_inc_norm",
    hue="optimizer_id",
    figure_filename: str = "figures/performance_over_time",
    figsize: tuple[int, int] = (6, 4),
    show_legend: bool = True,
    **lineplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    # TODO
    setup_seaborn(font_scale=1.5)
    sorter = get_order_by_mean(df=df)
    df = df.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))
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
