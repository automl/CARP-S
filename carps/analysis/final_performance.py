from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from carps.analysis.utils import get_color_palette, savefig, setup_seaborn


def plot_final_performance_boxplot(
        df: pd.DataFrame, x="n_trials_norm", y="trial_value__cost_inc_norm", hue="optimizer_id", 
        budget_var: str = "n_trials_norm", max_budget: float = 1,
        figure_filename: str = "figures/final_performance_boxplot.pdf", figsize: tuple[int,int]=(6,4), **boxplot_kwargs
    ) -> tuple[plt.Figure, matplotlib.axes.Axes]:

    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.boxplot(data=df[df[budget_var]==max_budget], y=y, x=x, hue=hue, palette=palette, ax=ax, **boxplot_kwargs)
    savefig(fig, figure_filename)
    return fig, ax