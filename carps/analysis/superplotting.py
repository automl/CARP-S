from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.core.utils import setup_globals
from rich.logging import RichHandler
import matplotlib.pyplot as plt


def setup_logging() -> None:
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name."""
    return logging.getLogger(logger_name)

setup_logging()
logger = get_logger(__file__)


setup_globals()

def normalize(S: pd.Series, epsilon: float = 1e-8) -> pd.Series:
    return (S - S.min()) / (S.max() - S.min() + epsilon)

def convert_mixed_types_to_str(logs: pd.DataFrame, logger=None) -> pd.DataFrame:
    mixed_type_columns = logs.select_dtypes(include=["O"]).columns
    if logger:
        logger.debug(f"Goodybe all mixed data, ruthlessly converting {mixed_type_columns} to str...")
    for c in mixed_type_columns:
        # D = logs[c]
        # logs.drop(columns=c)
        if c == "cfg_str":
            continue
        logs[c] = logs[c].map(lambda x: str(x))
        logs[c] = logs[c].astype("str")
    return logs


def normalize_logs(logs: pd.DataFrame) -> pd.DataFrame:
    logger.info("Start normalization...")
    logger.info("Normalize n_trials...")
    logs["n_trials_norm"] = logs.groupby("problem_id")["n_trials"].transform(normalize)
    logger.info("Normalize cost...")
    # Handle MO
    ids_mo = logs["scenario"]=="multi-objective"
    if len(ids_mo) > 0 and "hypervolume" in logs:
        hv = logs.loc[ids_mo, "hypervolume"]
        logs.loc[ids_mo, "trial_value__cost"] = -hv  # higher is better
        logs["trial_value__cost"] = logs["trial_value__cost"].astype("float64")
        logs["trial_value__cost_inc"] = logs["trial_value__cost"].transform("cummin")
    logs["trial_value__cost_norm"] = logs.groupby("problem_id")["trial_value__cost"].transform(normalize)
    logger.info("Calc normalized incumbent cost...")
    logs["trial_value__cost_inc_norm"] = logs.groupby(by=["problem_id", "optimizer_id", "seed"])["trial_value__cost_norm"].transform("cummin")
    if "time" not in logs:
        logs["time"] = 0
    logger.info("Normalize time...")
    logs["time_norm"] = logs.groupby("problem_id")["time"].transform(normalize)
    logs = convert_mixed_types_to_str(logs, logger)
    logger.info("Done.")
    return logs

def get_interpolated_performance_df(logs: pd.DataFrame, n_points: int = 20, x_column: str = "n_trials_norm", interpolation_columns: list[str] | None = None) -> pd.DataFrame:
    """Get performance dataframe for plotting.

    Interpolated at regular intervals.

    Parameters
    ----------
    logs : pd.DataFrame
        Preprocessed logs.
    n_points : int, optional
        Number of interpolation steps, by default 20
    x_column : str, optional
        The x-axis column to interpolate by, by default 'n_trials_norm'

    Raises:
    ------
    ValueError
        When x_column missing in dataframe.

    Returns:
    -------
    pd.DataFrame
        Performance data frame for plotting
    """
    if interpolation_columns is None:
        interpolation_columns = ["trial_value__cost", "trial_value__cost_norm", "trial_value__cost_inc", "trial_value__cost_inc_norm"]
    logger.info("Create dataframe for neat plotting by aligning x-axis / interpolating budget.")

    if x_column not in logs:
        msg = f"x_column `{x_column}` not in logs! Did you call `carps.analysis.process_data.process_logs` on the raw logs?"
        raise ValueError(msg)

    # interpolation_columns = [
    #     c for c in logs.columns if c != x_column and c not in identifier_columns and not c.startswith("problem")]
    group_keys = ["scenario", "set", "benchmark_id", "optimizer_id", "problem_id", "seed"]
    x = np.linspace(0, 1, n_points + 1)
    D = []
    for gid, gdf in logs.groupby(by=group_keys):
        metadata = dict(zip(group_keys, gid, strict=False))
        performance_data = {}
        performance_data[x_column] = x
        for icol in interpolation_columns:
            if icol in gdf:
                xp = gdf[x_column].to_numpy()
                fp = gdf[icol].to_numpy()
                y = np.interp(x=x, xp=xp, fp=fp)
                performance_data[icol] = y
        performance_data.update(metadata)
        D.append(pd.DataFrame(performance_data))
    return pd.concat(D).reset_index(drop=True)





def get_color_palette(df: pd.DataFrame | None) -> dict[str, Any]:
    cmap = sns.color_palette("colorblind", as_cmap=True)
    optimizers = list(df["optimizer_id"].unique())
    optimizers.sort()
    return {o: cmap[i] for i,o in enumerate(optimizers)}

def savefig(fig: plt.Figure, filename: str) -> None:
    figure_filename = Path(filename)
    figure_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_filename) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(figure_filename) + ".pdf", dpi=300, bbox_inches="tight")

def setup_seaborn(font_scale: float | None = None) -> None:
    if font_scale is not None:
        sns.set_theme(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

def filter_only_final_performance(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True) -> pd.DataFrame:
    if not soft:
        df = df[np.isclose(df[budget_var], max_budget)]
    else:
        df = df[df.groupby(["optimizer_id", "problem_id", "seed"])[budget_var].transform(lambda x: x == x.max())]
    return df


def plot_performance_over_time(df: pd.DataFrame, x="n_trials_norm", y="cost_inc_norm", hue="optimizer_id", figure_filename: str = "figures/performance_over_time.pdf", figsize: tuple[int,int]=(6,4), show_legend: bool = True, **lineplot_kwargs
                               ) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    setup_seaborn(font_scale=1.5)
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

def get_df_crit(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True, perf_col: str = "trial_value__cost_inc", remove_nan: bool = True) -> pd.DataFrame:
    df = filter_only_final_performance(df=df, budget_var=budget_var, max_budget=max_budget, soft=soft)

    # Work on mean of different seeds
    df_crit = df.groupby(["optimizer_id", "problem_id"])[perf_col].apply(np.nanmean).reset_index()

    df_crit = df_crit.pivot(
        index="problem_id",
        columns="optimizer_id",
        values=perf_col
    )

    if remove_nan:
        lost = df_crit[np.array([np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])]

        # Rows are problems, cols are optimizers
        df_crit = df_crit[np.array([not np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])]
        logger.info(f"Lost following experiments: {lost}")

    return df_crit


if __name__ == "__main__":
    # df.columns:
    # Index(['n_trials', 'n_incumbents', 'trial_value__cost',
    #        'trial_value__cost_inc', 'scenario', 'benchmark_id', 'problem_id',
    #        'optimizer_id', 'seed', 'reference_point', 'hypervolume', 'set'],
    #       dtype='object')
    # all but following necessary (but you can adjust also what the function works on): n_incumbents, scenario, reference_point, hypervolume
    df = pd.read_parquet("runs_subset_BB/dev/logs.parquet")
    df["set"] = "dev"

    df = normalize_logs(df)
    # print_overview(df)
    perf = get_interpolated_performance_df(df)
    perf_time = get_interpolated_performance_df(df, x_column="time_norm")



    lineplot_kwargs = {"linewidth": 3}
    for gid, gdf in perf.groupby(by=["scenario", "set"]):
        print(gid)
        fig, ax = plot_performance_over_time(
            df=gdf,
            x="n_trials_norm",
            y="trial_value__cost_inc_norm",
            hue="optimizer_id",
            figure_filename=f"figures/perf_over_time/performance_over_time_{gid}_trials.pdf",
            figsize=(6,4),
            **lineplot_kwargs
        )

        perf_col = "trial_value__cost_inc_norm"
        problem_prefix = ""
        fpath = Path("figures")
        fpath.mkdir(exist_ok=True, parents=True)
        identifier = gid

        # DF on normalized perf values
        df_crit = get_df_crit(gdf, remove_nan=False, perf_col=perf_col)
        # df_crit = df_crit.reindex(columns=names) # NOTE: Can order columns here
        df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
        df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
        plt.figure(figsize=(12, 12))
        sns.heatmap(df_crit, annot=False, fmt="g", cmap="viridis_r")
        plt.title("Performance of Optimizers per Problem (Normalized)")
        plt.ylabel("Problem ID")
        plt.xlabel("Optimizer")
        savefig(plt.gcf(), fpath / f"perf_opt_per_problem_{identifier}")
        plt.show()

        # Df on raw values
        # Optionally, plot the ranked data as a heatmap
        df_crit = get_df_crit(gdf, remove_nan=False, perf_col=perf_col)
        # df_crit = df_crit.reindex(columns=names) # NOTE: Can order columns here
        df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
        df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
        ranked_df = df_crit.rank(axis=1, method="min", ascending=True)

        plt.figure(figsize=(12, 12))
        sns.heatmap(ranked_df, annot=True, fmt="g", cmap="viridis_r")
        plt.title("Ranking of Optimizers per Problem")
        plt.ylabel("Problem ID")
        plt.xlabel("Optimizer")
        savefig(plt.gcf(), fpath / f"rank_opt_per_problem_{identifier}")
        plt.show()

        # Plotting the heatmap of the rank correlation matrix
        correlation_matrix = ranked_df.corr(method="spearman")
        plt.figure(figsize=(8,6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f")
        plt.title("Spearman Rank Correlation Matrix Between Optimizers")
        savefig(plt.gcf(), fpath / f"spearman_rank_corr_matrix_opt_{identifier}")
        plt.show()

    # Plot performance over actual time
    for gid, gdf in perf_time.groupby(by=["scenario", "set"]):
        print(gid)
        fig, ax = plot_performance_over_time(
            df=gdf,
            x="time_norm",
            y="trial_value__cost_inc_norm",
            hue="optimizer_id",
            figure_filename=f"figures/perf_over_time/performance_over_time_{gid}_elapsed.pdf",
            figsize=(6,4),
            **lineplot_kwargs
        )