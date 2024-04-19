from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from carps.analysis.utils import get_color_palette, savefig, setup_seaborn


def plot_performance_over_time(df: pd.DataFrame, x="n_trials_norm", y="cost_inc_norm", hue="optimizer_id", figure_filename: str = "figures/performance_over_time.pdf", figsize: tuple[int,int]=(6,4), **lineplot_kwargs
                               ) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, **lineplot_kwargs, ax=ax)
    savefig(fig, figure_filename)
    return fig, ax

def plot_rank_over_time(df: pd.DataFrame, x="n_trials_norm", y="cost_inc_norm", hue="optimizer_id", figure_filename: str = "figures/performance_over_time.pdf", figsize: tuple[int,int]=(6,4), **lineplot_kwargs
                               ) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    # TODO
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, **lineplot_kwargs, ax=ax)
    savefig(fig, figure_filename)
    return fig, ax