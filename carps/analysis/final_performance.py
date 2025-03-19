"""Analysis of final performance of optimizers."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from carps.analysis.utils import get_color_palette, savefig, setup_seaborn


def plot_final_performance_boxplot(
    df: pd.DataFrame,
    x: str = "n_trials_norm",
    y: str = "trial_value__cost_inc_norm",
    hue: str = "optimizer_id",
    budget_var: str = "n_trials_norm",
    max_fidelity: float = 1,
    figure_filename: str = "figures/final_performance_boxplot.pdf",
    figsize: tuple[int, int] = (6, 4),
    **boxplot_kwargs: dict,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot final performance as a boxplot.

    Args:
        df (pd.DataFrame): Dataframe with the logs.
        x (str, optional): x-axis column. Defaults to "n_trials_norm".
        y (str, optional): y-axis column. Defaults to "trial_value__cost_inc_norm".
        hue (str, optional): Hue column. Defaults to "optimizer_id".
        budget_var (str, optional): Budget variable. Defaults to "n_trials_norm". Necessary to get the final performance
            of the optimizers.
        max_fidelity (float, optional): Maximum budget. Defaults to 1.
        figure_filename (str, optional): Figure filename. Defaults to "figures/final_performance_boxplot.pdf".
        figsize (tuple[int, int], optional): Figure size. Defaults to (6, 4).
        **boxplot_kwargs: Additional boxplot arguments.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]: Figure and axes.
    """
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.boxplot(
        data=df[np.isclose(df[budget_var], max_fidelity)], y=y, x=x, hue=hue, palette=palette, ax=ax, **boxplot_kwargs
    )
    savefig(fig, figure_filename)
    return fig, ax


def plot_final_performance_violinplot(
    df: pd.DataFrame,
    x: str = "n_trials_norm",
    y: str = "trial_value__cost_inc_norm",
    hue: str = "optimizer_id",
    budget_var: str = "n_trials_norm",
    max_fidelity: float = 1,
    figure_filename: str = "figures/final_performance_boxplot.pdf",
    figsize: tuple[int, int] = (6, 4),
    **violinplot_kwargs: dict,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot final performance as a violinplot.

    Args:
        df (pd.DataFrame): Dataframe with the logs.
        x (str, optional): x-axis column. Defaults to "n_trials_norm".
        y (str, optional): y-axis column. Defaults to "trial_value__cost_inc_norm".
        hue (str, optional): Hue column. Defaults to "optimizer_id".
        budget_var (str, optional): Budget variable. Defaults to "n_trials_norm". Necessary to get the final performance
            of the optimizers.
        max_fidelity (float, optional): Maximum budget. Defaults to 1.
        figure_filename (str, optional): Figure filename. Defaults to "figures/final_performance_boxplot.pdf".
        figsize (tuple[int, int], optional): Figure size. Defaults to (6, 4).
        **violinplot_kwargs: Additional violinplot arguments.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]: Figure and axes.
    """
    setup_seaborn()
    palette = get_color_palette(df)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.violinplot(
        data=df[np.isclose(df[budget_var], max_fidelity)],
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


def create_tables(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_fidelity: float = 1) -> None:
    """Create tables for final performance.

    Might be unfinished?

    Args:
        df (pd.DataFrame): Dataframe with the logs.
        budget_var (str, optional): Budget variable. Defaults to "n_trials_norm".
        max_fidelity (float, optional): Maximum budget. Defaults
    """
    perf_col_norm: str = "trial_value__cost_inc_norm"

    print(df[budget_var].max())
    df = df[np.isclose(df[budget_var], max_fidelity)]  # noqa: PD901

    # Aggregate all

    # Calculate mean over seeds per optimizer and task
    df_mean = df.groupby(["optimizer_id", "task_id"])[perf_col_norm].mean()
    df_mean.name = "mean"
    df_var = df.groupby(["optimizer_id", "task_id"])[perf_col_norm].var()
    df_var.name = "var"

    print(pd.concat((df_mean, df_var), axis=1))

    # Calculate mean over tasks

    # Aggregate over benchmarks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from carps.analysis.process_data import load_logs

    rundir = "/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/runs"

    df, df_cfg = load_logs(rundir=rundir)
    create_tables(df)
