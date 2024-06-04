from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from carps.analysis.utils import get_color_palette, savefig, setup_seaborn


def plot_final_performance_boxplot(
    df: pd.DataFrame,
    x="n_trials_norm",
    y="trial_value__cost_inc_norm",
    hue="optimizer_id",
    budget_var: str = "n_trials_norm",
    max_budget: float = 1,
    figure_filename: str = "figures/final_performance_boxplot.pdf",
    figsize: tuple[int, int] = (6, 4),
    **boxplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.boxplot(
        data=df[np.isclose(df[budget_var], max_budget)], y=y, x=x, hue=hue, palette=palette, ax=ax, **boxplot_kwargs
    )
    savefig(fig, figure_filename)
    return fig, ax


def plot_final_performance_violinplot(
    df: pd.DataFrame,
    x="n_trials_norm",
    y="trial_value__cost_inc_norm",
    hue="optimizer_id",
    budget_var: str = "n_trials_norm",
    max_budget: float = 1,
    figure_filename: str = "figures/final_performance_boxplot.pdf",
    figsize: tuple[int, int] = (6, 4),
    **violinplot_kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.violinplot(
        data=df[np.isclose(df[budget_var], max_budget)],
        y=y,
        x=x,
        hue=hue,
        palette=palette,
        ax=ax,
        cut=0,
        **violinplot_kwargs,
    )
    savefig(fig, figure_filename)
    return fig, ax


def create_tables(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1):
    perf_col_norm: str = "trial_value__cost_inc_norm"

    print(df[budget_var].max())
    df = df[np.isclose(df[budget_var], max_budget)]

    # Aggregate all

    # Calculate mean over seeds per optimizer and problem
    df_mean = df.groupby(["optimizer_id", "problem_id"])[perf_col_norm].mean()
    df_mean.name = "mean"
    df_var = df.groupby(["optimizer_id", "problem_id"])[perf_col_norm].var()
    df_var.name = "var"

    print(pd.concat((df_mean, df_var), axis=1))

    # Calculate mean over problems

    # Aggregate over benchmarks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from carps.analysis.process_data import load_logs

    rundir = "/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/runs"

    df, df_cfg = load_logs(rundir=rundir)
    create_tables(df)
