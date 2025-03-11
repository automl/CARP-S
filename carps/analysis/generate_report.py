"""Generate a report from the results of the optimization runs.

For this we need the run logs, which are stored in a CSV or parquet file.

To generate this file, call `python -m carps.analysis.gather_results <rundir>`.
Of course, you can concatenate multiple result files.
Keep in mind that the results are grouped by task_type and set id.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from seaborn.utils import (
    relative_luminance,
)

from carps.analysis.gather_data import (
    get_interpolated_performance_df,
    normalize_logs,
)
from carps.analysis.run_autorank import (
    calc_critical_difference,
    cd_evaluation,
    get_df_crit,
    get_sorted_rank_groups,
)
from carps.analysis.utils import filter_only_final_performance, get_color_palette, savefig, setup_seaborn
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


def _annotate_heatmap(
    ax: matplotlib.axes.Axes, mesh: matplotlib.collections.QuadMesh, annot_kws: dict, fmt: str, annot_data: np.ndarray
) -> None:
    """Add textual labels with the value in each cell.

    Takes care of the color contrast between the text and the cell.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        mesh (matplotlib.collections.QuadMesh): The mesh of the heatmap.
        annot_kws (dict): Keyword arguments for the annotation.
        fmt (str): The format string.
        annot_data (np.ndarray): The data to annotate.
    """
    mesh.update_scalarmappable()
    height, width = annot_data.shape
    xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    for x, y, m, color, val in zip(
        xpos.flat, ypos.flat, mesh.get_array().flat, mesh.get_facecolors(), annot_data.flat, strict=False
    ):
        if m is not np.ma.masked:
            lum = relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"  # noqa: PLR2004
            annotation = ("{:" + fmt + "}").format(val)
            text_kwargs = {"color": text_color, "ha": "center", "va": "center"}
            text_kwargs.update(annot_kws)
            ax.text(x, y, annotation, **text_kwargs)


def plot_ranks_over_time(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the ranks of the optimizers over time.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.3)
    lineplot_kwargs = {"linewidth": 4}

    key_performance = "trial_value__cost_inc"
    x_column = "n_trials_norm"

    perf = get_interpolated_performance_df(df)

    df_rank_list = []
    for gid, gdf in perf.groupby(["task_type", "set"]):
        budgets = gdf[x_column].unique()
        for max_fidelity in budgets:
            df_crit = get_df_crit(
                gdf, max_fidelity=max_fidelity, perf_col=key_performance, budget_var=x_column, soft=False
            )
            rank_result = cd_evaluation(
                df_crit,
                maximize_metric=False,
                ignore_non_significance=True,
                plot_diagram=False,
            )
            df_rank_list.append(
                pd.DataFrame(
                    {
                        "task_type": gid[0],
                        "set": gid[1],
                        "optimizer_id": rank_result.rankdf.index,
                        "n_trials_norm": max_fidelity,
                        "rank": rank_result.rankdf["meanrank"],
                    }
                )
            )
    df_rank: pd.DataFrame = pd.concat(df_rank_list).reset_index(drop=True)
    df_rank.to_csv("df_rank.csv", index=False)

    resulting_files = []
    for gid, gdf in df_rank.groupby(["task_type", "set"]):
        palette = get_color_palette(gdf)
        figure_filename = f"{output_dir}/rank_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename,
                "plot_type": "rank_over_time",
                "plot_type_pretty": "Rank over Time",
                "explanation": "The rank of each optimizer over time compares which optimizer "
                "performs better, the lower "
                "the rank the better. For each optimizer and task, the performance is averaged over seeds to obtain"
                " an estimate of the performance. The rank is then calculated per step and task with the same "
                "approach as for the critical difference diagram.",
            }
        )
        if not replot:
            continue

        palette = get_color_palette(gdf)
        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax = sns.lineplot(data=gdf, x=x_column, y="rank", hue="optimizer_id", ax=ax, palette=palette, **lineplot_kwargs)
        ax.set_xlabel("Number of Trials (normalized)")
        ax.set_ylabel("Rank (lower is better)")
        ax.set_xlim(0, 1)

        final_rank = gdf[gdf[x_column] == df_rank[x_column].max()].groupby("optimizer_id")["rank"].mean().sort_values()  # type:ignore[[call-overload]
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels, strict=False), key=lambda x: final_rank[x[1]])
        sorted_handles, sorted_labels = zip(*sorted_handles_labels, strict=False)
        sorted_labels = tuple([f"{label} ({final_rank[label]:.1f})" for label in sorted_labels])
        legend_title = "Optimizer (Final Rank)"
        ax.legend(sorted_handles, sorted_labels, loc="center left", bbox_to_anchor=(1.05, 0.5), title=legend_title)
        ax.set_title(f"Scenario: {gid[0]}, Set: {gid[1]}")
        savefig(fig, figure_filename)
        plt.close(fig)

    return resulting_files


def plot_ecdf(df: pd.DataFrame, output_dir: str | Path = "figures", replot: bool = True) -> list[dict[str, Any]]:  # noqa: FBT001, FBT002
    """Plot the empirical cumulative distribution function (eCDF) / proportion of the incumbent cost.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.3)
    lineplot_kwargs = {"linewidth": 3}

    key_performance = "trial_value__cost_inc_log_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        palette = get_color_palette(gdf)
        figure_filename = f"{output_dir}/ecdf_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename,
                "plot_type": "ecdf",
                "plot_type_pretty": "Proportion of Incumbent Cost",
                "explanation": "The empirical cumulative distribution function (eCDF) shows the observed "
                "distribution of the "
                "incumbent cost over time. The incumbent costs are first logarithmized and then normalized. "
                "The eCDF shows the proportion of incumbent costs encountered during the optimization. "
                "The further left the curve is, the better the optimizer is performing, because it achieves lower "
                "values sooner.",
            }
        )
        if not replot:
            continue

        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        for optimizer_id, odf in gdf.groupby("optimizer_id"):
            ax = sns.ecdfplot(
                data=odf, x=key_performance, ax=ax, label=optimizer_id, color=palette[optimizer_id], **lineplot_kwargs
            )
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        # ax.set_xscale("log")
        ax.set_xlabel("Log Incumbent Cost (Normalized)")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{gid[0]}: {gid[1]}")
        savefig(fig, figure_filename)
        plt.close(fig)

    return resulting_files


def plot_critical_difference(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the critical difference diagram.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    perf_col: str = "trial_value__cost_inc"
    figsize = (6 * 1.5, 4 * 1.5)

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        fig_filename = f"{output_dir}/criticaldifference_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": fig_filename,
                "plot_type": "critical_difference",
                "plot_type_pretty": "Critical Difference",
                "explanation": "Most importantly, we propose statistical testing on rankings as our main result. "
                r"For the ranking, we use the library \code{autorank}~\citep{herbold-joss20} for "
                "determining the ranks and critical differences. "
                "The ranking is performed on the raw performance values, averaged across seeds. "
                r"To be more precise, we use the frequentist approach~\citep{demsar-06a}: We use the non-parametric "
                "Friedman test as an omnibus test to determine whether there are any significant differences between "
                "the median values of the populations. "
                "We use this test because we have more than two populations, which cannot be assumed to be normally "
                "distributed. "
                "We use the post hoc Nemenyi test to infer which differences are significant. "
                "The significance level is $\alpha=0.05$. "
                "In order to be considered different, the difference between the mean ranks of two optimizers must be "
                "greater than the critical difference (CD).",
            }
        )
        if not replot:
            continue
        df_crit = get_df_crit(gdf, perf_col=perf_col)
        _ = cd_evaluation(
            df_crit,
            maximize_metric=False,
            ignore_non_significance=True,
            output_path=fig_filename,
            figsize=figsize,
            plot_diagram=True,
        )
    return resulting_files


def plot_performance_per_task(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the performance of the optimizers per task.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.3)

    perf_col = "trial_value__cost_inc_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        figure_filename = f"{output_dir}/performancepertask_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename,
                "plot_type": "performance_per_task",
                "plot_type_pretty": "Performance per Task",
                "explanation": "The heatmap shows the performance of the optimizers per task. "
                "The performance is first log-transformed, then normalized and averaged over seeds. "
                "The performance is shown as a heatmap, where the colors indicate the performance of the optimizer "
                "on a specific task. "
                "The better the optimizer performs, the lighter/more yellow the color.",
            }
        )
        if not replot:
            continue

        # result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)

        # sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)

        fig = plt.Figure(figsize=(12, 12))
        ax0 = fig.add_subplot(111)

        # Perf per task (normalized)
        df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
        offset = 1e-8
        df_crit[df_crit == 0] = offset
        ax0 = sns.heatmap(
            df_crit,
            annot=False,
            fmt=".6f",
            cmap="viridis_r",
            ax=ax0,
            cbar_kws={"shrink": 0.8, "aspect": 30},
            norm=LogNorm(vmin=offset, vmax=1),
            annot_kws={"fontsize": 8},
        )
        df_crit_raw = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col.replace("_norm", ""))
        annot_data = df_crit_raw.to_numpy()

        mesh = ax0.collections[0]
        _annotate_heatmap(ax0, mesh, {"fontsize": 8}, ".6g", annot_data)

        ax0.set_title(
            f"Final Performance per Task for Scenario {gid[0]} and Set {gid[1]}\n"
            "Annotations: Raw Values, Colormap: Normalized Values"
        )

        ax0.set_title(f"Final Performance per Task for Scenario {gid[0]} and Set {gid[1]}")
        ax0.text(
            0.5,
            1.05,
            "Annotations: Raw Values, Colormap: Normalized Values",
            ha="center",
            va="bottom",
            fontsize=12,
            transform=ax0.transAxes,
        )

        ax0.set_ylabel("Task ID")
        ax0.set_xlabel("Optimizer")
        savefig(fig, figure_filename)
        plt.close(fig)
    return resulting_files


def plot_boxplot_violinplot(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the final performance of the optimizers as boxplot and violinplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.3)

    perf_col = "trial_value__cost_inc_log_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        palette = get_color_palette(gdf)
        figure_filename_boxplot = f"{output_dir}/finalperfboxplot_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename_boxplot,
                "plot_type": "finalperformance_boxplot",
                "plot_type_pretty": "Final Performance (Normalized, Boxplot)",
                "explanation": "The boxplot shows the final performance of the optimizers. "
                "The performance is first log-transformed, then normalized and averaged over seeds. ",
            }
        )
        figure_filename_violinplot = f"{output_dir}/finalperfviolinplot_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename_violinplot,
                "plot_type": "finalperformance_violinplot",
                "plot_type_pretty": "Final Performance (Normalized, Violinplot)",
                "explanation": "The violinplot shows the final performance of the optimizers. "
                "The performance is first log-transformed, then normalized and averaged over seeds. ",
            }
        )
        if not replot:
            continue

        result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)
        sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)

        fig = plt.Figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        df_finalperf = filter_only_final_performance(df=gdf)
        sorter = names
        df_finalperf = df_finalperf.sort_values(
            by="optimizer_id",
            key=lambda column: column.map(lambda e: sorter.index(e)),  # noqa: B023
        )
        palette = get_color_palette(df=gdf)
        x = x_column
        y = perf_col
        x = y
        hue = "optimizer_id"
        y = hue
        ax1 = sns.boxplot(data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax1)
        ax1.set_title("Log Final Performance (Normalized)")
        savefig(fig, figure_filename_boxplot)
        plt.close(fig)

        fig = plt.Figure(figsize=(6, 4))
        ax2 = fig.add_subplot(111)
        ax2 = sns.violinplot(data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax2, cut=0)
        ax2.set_title("Log Final Performance (Normalized)")
        savefig(fig, figure_filename_violinplot)
        plt.close(fig)

    return resulting_files


def plot_finalperfbarplot(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the final performance of the optimizers as boxplot and violinplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.2)

    perf_col = "trial_value__cost_inc_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        palette = get_color_palette(gdf)
        figure_filename = f"{output_dir}/finalperfbarplot_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename,
                "plot_type": "finalperformance_barplot",
                "plot_type_pretty": "Final Performance (Normalized, Barplot)",
                "explanation": "The barplot shows the mean final performance of the optimizers "
                r"with 95-\% confidence interval.",
            }
        )
        if not replot:
            continue

        result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)
        sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)

        fig = plt.Figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        df_finalperf = filter_only_final_performance(df=gdf)
        df_finalperf["mean_perf"] = df_finalperf.groupby("optimizer_id")[perf_col].transform("mean")
        df_finalperf = df_finalperf.sort_values(by="mean_perf")
        # df_finalperf = df_finalperf.sort_values(
        #     by="optimizer_id",
        #     key=lambda column: column.map(lambda e: sorter.index(e)),
        # )
        palette = get_color_palette(df=gdf)
        x = x_column
        y = perf_col
        x = y
        hue = "optimizer_id"
        y = hue
        ax1 = sns.barplot(data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax1)
        ax1.set_title("Final Performance (Normalized)")
        ax1.set_xscale("log")
        savefig(fig, figure_filename)
        plt.close(fig)

    return resulting_files


def plot_spearman_rank_correlation(
    df: pd.DataFrame,
    output_dir: str | Path = "figures",
    replot: bool = True,  # noqa: FBT001, FBT002
) -> list[dict[str, Any]]:
    """Plot the Spearman rank correlation matrix between the optimizers.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        output_dir (str | Path, "figures"): The output directory to save the plots to.
        replot (bool, True): Whether to replot the figures.

    Returns:
        list[dict[str, Any]]: The filenames of and information about the resulting plots.
    """
    setup_seaborn(font_scale=1.2)

    perf_col = "trial_value__cost_inc_log_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["task_type", "set"]):
        figure_filename = f"{output_dir}/spearmanrankcorrelation_{gid[0]}_{gid[1]}"
        resulting_files.append(
            {
                "task_type": gid[0],
                "set": gid[1],
                "task_id": None,
                "filename": figure_filename,
                "plot_type": "spearman_rank_correlation",
                "plot_type_pretty": "Spearman Rank Correlation",
                "explanation": "The Spearman rank correlation matrix shows the correlation between the "
                "ranks of the optimizers. "
                "The intuition is that optimizers that perform similarly on the tasks will have a high correlation. "
                "The ranks are calculated based on the final performance of the optimizers. ",
            }
        )
        if not replot:
            continue

        result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)
        sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)
        df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
        df_crit = df_crit.reindex(columns=names)
        # df_crit.index = [i.replace(task_prefix + "/dev/", "") for i in df_crit.index]
        # df_crit.index = [i.replace(task_prefix + "/test/", "") for i in df_crit.index]

        fig = plt.Figure(figsize=(6 * 1.5, 4 * 1.5))
        ax3 = fig.add_subplot(111)
        ranked_df = df_crit.rank(axis=1, method="min", ascending=True)
        correlation_matrix = ranked_df.corr(method="spearman")
        ax3 = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f", ax=ax3)
        ax3.set_title("Spearman Rank Correlation Matrix\nBetween Optimizers")

        # fig.set_tight_layout(True)

        savefig(fig, figure_filename)

        plt.close(fig)

    return resulting_files


def load_results(result_path: str) -> pd.DataFrame:
    """Load and preprocess the results of the optimization runs.

    Parameters
    ----------
    result_path : str
        The path to results, can be a CSV or parquet file.

    Returns.
    -------
    pd.DataFrame
        The preprocessed results.
    """
    # 1. Load results
    logger.info("Loading results")
    df = pd.read_parquet(result_path) if result_path.endswith(".parquet") else pd.read_csv(result_path)  # noqa: PD901

    # 2. Preprocess results
    logger.info("Preprocessing results")
    print(df.columns)

    df = normalize_logs(df)  # noqa: PD901
    if "set" not in df.columns:
        df["set"] = df["task_id"].apply(lambda x: "dev" if "dev" in x else "test")
    return df


latex_template_explanation_block = r"""\subsubsection{explanation_title}
explanation_text

"""

latex_template_plot_block = r"""\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\textwidth]{plot_filename}
    \caption{plot_title}
    \label{fig:plot_filename}
\end{figure}

"""

full_report_template = r"""\documentclass{article}
\usepackage{graphicx}

\begin{document}

\title{Report}

report_tex

\end{document}
"""


def write_latex_report(resulting_files: pd.DataFrame, report_dir: str | Path, report_name: str) -> None:
    """Write latex report.

    Parameters
    ----------
    resulting_files : pd.DataFrame
        The filenames and information of the resulting plots.
    report_dir : str
        The directory to save the report to.
    report_name : str
        The name of the report.
    """
    report_dir = Path(report_dir)
    order = {
        "Final Performance": [
            "critical_difference",
            # "spearman_rank_correlation",
            "performance_per_task",
            # "finalperformance_boxplot",
            # "finalperformance_violinplot",
            "finalperformance_barplot",
        ],
        "Anytime Performance": [
            "rank_over_time",
            # "ecdf",
        ],
    }

    for (task_type, set_id), info in resulting_files.groupby(["task_type", "set"]):
        report_tex = ""
        report_filename = report_dir / f"{report_name}_{task_type}_{set_id}.tex"
        full_report_filename = report_dir / f"full_{report_name}_{task_type}_{set_id}.tex"

        print(task_type, set_id)

        # Embed plots
        report_tex += "\\section{Plots}\n"
        for group_name, _order in order.items():
            report_tex += f"\\subsection{{{group_name}}}\n"

            for plot_type in _order:
                _info = info[info["plot_type"] == plot_type].iloc[0]
                plot_title = f"Scenario: {_info['task_type']} - Set: {_info['set']} - {_info['plot_type_pretty']}"
                plot_filename = "figures" + _info["filename"].split("figures")[-1]
                report_tex += latex_template_plot_block.replace("plot_title", plot_title).replace(
                    "plot_filename", plot_filename
                )

        # Add explanation
        report_tex += "\\section{Explanation of Plots}\n"
        for group_name, _order in order.items():
            report_tex += f"\\subsection{{{group_name}}}\n"
            for plot_type in _order:
                _info = info[info["plot_type"] == plot_type].iloc[0]
                report_tex += latex_template_explanation_block.replace(
                    "explanation_title", _info["plot_type_pretty"]
                ).replace("explanation_text", _info["explanation"])
        print(report_tex)
        break

    with open(report_filename, "w") as f:
        f.write(report_tex)

    full_report_tex = full_report_template.replace("report_tex", report_tex)
    with open(full_report_filename, "w") as f:
        f.write(full_report_tex)


def generate_report(
    result_path: str = "logs.parquet",
    report_dir: str | Path = "reports",
    report_name: str = "report",
) -> None:
    """Generate a report from the results of the optimization runs.

    Args:
        result_path (str, "logs.parquet"): Path to the results CSV or parquet file.
        report_dir (str | Path, "reports"): Directory to save the report to.
        report_name (str, "report"): Name of the report.
    """
    logger.info("Generating report")

    date_and_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    date_and_time = "const"

    report_dir = Path(report_dir)
    report_dir = report_dir / date_and_time
    report_dir.mkdir(exist_ok=True, parents=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)

    # Load and preprocess results
    df = load_results(result_path)  # noqa: PD901

    # FINAL PERFORMANCE
    logger.info("Plotting final performance...")

    # Critical Difference
    logger.info("\t...critical difference")
    resulting_files_critical_difference = plot_critical_difference(df, output_dir=figure_dir, replot=True)

    # Final Performance per Task (Mean over seeds, heatmap)
    logger.info("\t...performance per task")
    resulting_files_performance_per_task = plot_performance_per_task(df, output_dir=figure_dir, replot=True)

    # Final Performance Barplot per Task (Mean over seeds with std)
    logger.info("\t...barplot")
    resulting_files_finalperfbarplot = plot_finalperfbarplot(df, output_dir=figure_dir, replot=True)

    # ANYTIME PERFORMANCE
    logger.info("Plotting anytime performance...")

    # Plot ranks over time
    logger.info("\t...ranks over time")
    resulting_files_rank_over_time = plot_ranks_over_time(df, output_dir=figure_dir, replot=True)

    resulting_files = pd.concat(
        [
            pd.DataFrame(resulting_files_critical_difference),
            pd.DataFrame(resulting_files_performance_per_task),
            pd.DataFrame(resulting_files_finalperfbarplot),
            pd.DataFrame(resulting_files_rank_over_time),
        ]
    ).reset_index(drop=True)
    write_latex_report(resulting_files, report_dir, report_name)


if __name__ == "__main__":
    fire.Fire(generate_report)
