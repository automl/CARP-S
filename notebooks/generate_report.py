from __future__ import annotations
import fire
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
# from carps.analysis.process_data import get_interpolated_performance_df, load_logs, process_logs
import importlib
import carps
import carps.analysis
import carps.analysis.gather_data
from carps.analysis.gather_data import normalize_logs, get_interpolated_performance_df, load_logs, process_logs, load_set
from carps.analysis.utils import get_color_palette, savefig, setup_seaborn, filter_only_final_performance
import seaborn as sns

from carps.analysis.run_autorank import calc_critical_difference, custom_latex_table, get_df_crit, cd_evaluation, get_sorted_rank_groups
from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.task import Task
from carps.utils.trials import TrialInfo

setup_logging()
logger = get_logger(__file__)


def plot_ranks_over_time(perf: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    setup_seaborn(font_scale=1.3)
    palette = get_color_palette(perf)
    lineplot_kwargs = dict(linewidth=3)

    key_performance = "trial_value__cost_inc"
    x_column = "n_trials_norm"

    # Calculate the rank of each optimizer for each problem. The estimated performance is
    # the mean of all seeds. We use the interpolated performance, otherwise it is not
    # possible to plot well with seaborn (seaborn needs a value for each x value).
    group_keys_estimate = ["scenario", "set", "benchmark_id", "problem_id", "optimizer_id", x_column]
    df_estimated = perf.groupby(group_keys_estimate)[key_performance].mean().reset_index()
    df_rank = df_estimated.copy()
    df_rank["rank"] = df_estimated.groupby(["scenario", "set", "benchmark_id", "problem_id", x_column])[key_performance].rank(ascending=True, method="min")

    resulting_files = []
    for gid, gdf in df_rank.groupby(["scenario", "set"]):
        figure_filename = f"{output_dir}/rank_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename,
            "plot_type": "rank_over_time",
            "plot_type_pretty": "Rank over Time",
            "explanation": "The rank of each optimizer over time compares which optimizer performs better, the lower "\
                "the rank the better. For each optimizer and problem, the performance is averaged over seeds to obtain"\
                " an estimate of the performance. The rank is then calculated per step and problem.",
        })
        if not replot:
            continue

        final_rank = gdf[gdf[x_column] == df_rank[x_column].max()].groupby("optimizer_id")["rank"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax = sns.lineplot(data=gdf, x=x_column, y="rank", hue="optimizer_id", ax=ax, palette=palette, **lineplot_kwargs)
        ax.set_xlabel("Number of Trials (normalized)")
        ax.set_ylabel("Rank (lower is better)")
        ax.set_xlim(0, 1)
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: final_rank[x[1]])
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        sorted_labels = [f"{l} ({final_rank[l]:.1f})" for l in sorted_labels]
        ax.legend(sorted_handles, sorted_labels, loc="center left", bbox_to_anchor=(1.05, 0.5))
        # ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax.set_title(f"{gid[0]}: {gid[1]}")
        savefig(fig, figure_filename)
        plt.show()

    return resulting_files

def plot_ecdf(df: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    setup_seaborn(font_scale=1.3)
    palette = get_color_palette(df)
    lineplot_kwargs = dict(linewidth=3)

    key_performance = "trial_value__cost_inc_log_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["scenario", "set"]):
        figure_filename = f"{output_dir}/ecdf_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename,
            "plot_type": "ecdf",
            "plot_type_pretty": "Proportion of Incumbent Cost",
            "explanation": "The empirical cumulative distribution function (eCDF) shows the observed distribution of the "\
                "incumbent cost over time. The incumbent costs are first logarithmized and then normalized. "\
                "The eCDF shows the proportion of incumbent costs encountered during the optimization. "\
                "The further left the curve is, the better the optimizer is performing, because it achieves lower "\
                "values sooner."
        })
        if not replot:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        for optimizer_id, odf in gdf.groupby("optimizer_id"):
            ax = sns.ecdfplot(data=odf, x=key_performance, ax=ax, label=optimizer_id, color=palette[optimizer_id], **lineplot_kwargs)
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        # ax.set_xscale("log")
        ax.set_xlabel("Log Incumbent Cost (Normalized)")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{gid[0]}: {gid[1]}")
        savefig(fig, figure_filename)
        plt.show()

    return resulting_files

def plot_critical_difference(df: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    perf_col: str = "trial_value__cost_inc_log_norm"
    figsize = (6 * 1.5, 4 * 1.5)

    resulting_files = []
    for gid, gdf in df.groupby(["scenario", "set"]):
        fig_filename = f"{output_dir}/criticaldifference_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": fig_filename,
            "plot_type": "critical_difference",
            "plot_type_pretty": "Critical Difference",
            "explanation": "Most importantly, we propose statistical testing on rankings as our main result. "\
                "For the ranking, we use the library \code{autorank}~\citep{herbold-joss20} for determining the ranks and critical differences. "\
                "The ranking is performed on the raw performance values, averaged across seeds. "\
                "To be more precise, we use the frequentist approach~\citep{demsar-06a}: We use the non-parametric Friedman test as an omnibus test to determine whether there are any significant differences between the median values of the populations. "\
                "We use this test because we have more than two populations, which cannot be assumed to be normally distributed. "\
                "We use the post hoc Nemenyi test to infer which differences are significant. "\
                "The significance level is $\alpha=0.05$. "\
                "In order to be considered different, the difference between the mean ranks of two optimizers must be greater than the critical difference (CD).",
        })
        if not replot:
            continue
        df_crit = get_df_crit(df, perf_col=perf_col)
        rank_result = cd_evaluation(
            df_crit,
            maximize_metric=False,
            ignore_non_significance=True,
            output_path=fig_filename,
            figsize=figsize,
            plot_diagram=True,
        )
    return resulting_files

def plot_performance_per_problem(df: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    setup_seaborn(font_scale=1.3)
    palette = get_color_palette(df)
    lineplot_kwargs = dict(linewidth=3)

    perf_col = "trial_value__cost_inc_log_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["scenario", "set"]):
        figure_filename = f"{output_dir}/performanceperproblem_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename,
            "plot_type": "performance_per_problem",
            "plot_type_pretty": "Performance per Problem",
            "explanation": "The heatmap shows the performance of the optimizers per problem. "\
                "The performance is first log-transformed, then normalized and averaged over seeds. "\
                "The performance is shown as a heatmap, where the colors indicate the performance of the optimizer on a specific problem. "\
                "The better the optimizer performs, the lighter/more yellow the color.",
        })
        if not replot:
            continue

        # result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)

        # sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)


        fig, ax0 = plt.subplots(figsize=(12, 12))

        # Perf per problem (normalized)
        df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
        # df_crit = df_crit.reindex(columns=names)
        # df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
        # df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
        ax0 = sns.heatmap(df_crit, annot=False, fmt="g", cmap="viridis_r", ax=ax0, cbar_kws={"shrink": 0.8, "aspect": 30})
        ax0.set_title("Log Final Performance per Problem (Normalized)")
        ax0.set_ylabel("Problem ID")
        ax0.set_xlabel("Optimizer")
        savefig(fig, figure_filename)
        plt.show()
    return resulting_files

def plot_boxplot_violinplot(df: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    setup_seaborn(font_scale=1.3)
    palette = get_color_palette(df)
    lineplot_kwargs = dict(linewidth=3)

    perf_col = "trial_value__cost_inc_log_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["scenario", "set"]):
        figure_filename_boxplot = f"{output_dir}/finalperfboxplot_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename_boxplot,
            "plot_type": "finalperformance_boxplot",
            "plot_type_pretty": "Final Performance (Normalized, Boxplot)",
            "explanation": "The boxplot shows the final performance of the optimizers. "\
                "The performance is first log-transformed, then normalized and averaged over seeds. "\
        })
        figure_filename_violinplot = f"{output_dir}/finalperfviolinplot_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename_violinplot,
            "plot_type": "finalperformance_violinplot",
            "plot_type_pretty": "Final Performance (Normalized, Violinplot)",
            "explanation": "The violinplot shows the final performance of the optimizers. "\
                "The performance is first log-transformed, then normalized and averaged over seeds. "\
        })
        if not replot:
            continue

        result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)
        sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)

        fig, ax1 = plt.subplots(figsize=(6, 4))
        df_finalperf = filter_only_final_performance(df=gdf)
        sorter = names
        df_finalperf = df_finalperf.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))
        palette = get_color_palette(df=gdf)
        x=x_column
        y=perf_col
        x = y
        hue="optimizer_id"
        y = hue
        ax1 = sns.boxplot(
            data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax1
        )
        ax1.set_title("Log Final Performance (Normalized)")
        savefig(fig, figure_filename_boxplot)
        plt.show()

        fig, ax2 = plt.subplots(figsize=(6, 4))
        ax2 = sns.violinplot(
            data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax2, cut=0
        )
        ax2.set_title("Log Final Performance (Normalized)")
        savefig(fig, figure_filename_violinplot)
        plt.show()

    return resulting_files

def plot_spearman_rank_correlation(df: pd.DataFrame, output_dir: str = "figures", replot: bool = True) -> dict[tuple[str, str], str]:
    setup_seaborn(font_scale=1.2)
    palette = get_color_palette(df)
    lineplot_kwargs = dict(linewidth=3)

    perf_col = "trial_value__cost_inc_log_norm"
    x_column = "n_trials_norm"

    resulting_files = []
    for gid, gdf in df.groupby(["scenario", "set"]):
        figure_filename = f"{output_dir}/spearmanrankcorrelation_{gid[0]}_{gid[1]}"
        resulting_files.append({
            "scenario": gid[0],
            "set": gid[1],
            "problem_id": None,
            "filename": figure_filename,
            "plot_type": "spearman_rank_correlation",
            "plot_type_pretty": "Spearman Rank Correlation",
            "explanation": "The Spearman rank correlation matrix shows the correlation between the ranks of the optimizers. "\
                "The intuition is that optimizers that perform similarly on the problems will have a high correlation. "\
                "The ranks are calculated based on the final performance of the optimizers. ",
        })
        if not replot:
            continue

        result = calc_critical_difference(gdf, identifier=None, perf_col=perf_col, plot_diagram=False)
        sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)
        df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
        df_crit = df_crit.reindex(columns=names)
        # df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
        # df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]

        fig, ax3 = plt.subplots(figsize=(6 * 1.5, 4 * 1.5))
        ranked_df = df_crit.rank(axis=1, method="min", ascending=True)
        correlation_matrix = ranked_df.corr(method="spearman")
        ax3 = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f", ax=ax3)
        ax3.set_title("Spearman Rank Correlation Matrix\nBetween Optimizers")

        # fig.set_tight_layout(True)

        savefig(fig, figure_filename)

        plt.show()

    return resulting_files

def generate_report(
    result_csv_path: str = "results.csv",
    report_dir: str = "reports",
    report_name: str = "report",
):
    logger.info("Generating report")

    date_and_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    date_and_time = "const"

    report_dir = Path(report_dir) / date_and_time
    report_dir.mkdir(exist_ok=True, parents=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)


    # # 1. Load results
    # logger.info("Loading results")
    # df = pd.read_csv(result_csv_path)

    # # 2. Preprocess results
    # logger.info("Preprocessing results")
    # print(df.columns)
    # df = normalize_logs(df)
    # perf = get_interpolated_performance_df(df)

    # # FINAL PERFORMANCE
    # logger.info("Plotting final performance...")

    # # Critical Difference
    # logger.info("\t...critical difference")
    # resulting_files_critical_difference = plot_critical_difference(df, output_dir=figure_dir, replot=False)

    # # Spearman Rank Correlation
    # logger.info("\t...spearman rank correlation")
    # resulting_files_spearman_rank_correlation = plot_spearman_rank_correlation(df, output_dir=figure_dir, replot=False)

    # # Final Performance per Problem (Mean over seeds, heatmap)
    # logger.info("\t...performance per problem")
    # resulting_files_performance_per_problem = plot_performance_per_problem(df, output_dir=figure_dir, replot=False)

    # # Final Performance as boxplot and violinplot
    # logger.info("\t...boxplot and violinplot")
    # resulting_files_boxplot_violinplot = plot_boxplot_violinplot(df, output_dir=figure_dir, replot=False)

    # # ANYTIME PERFORMANCE
    # logger.info("Plotting anytime performance...")

    # # Plot ranks over time
    # logger.info("\t...ranks over time")
    # resulting_files_rank_over_time = plot_ranks_over_time(perf, output_dir=figure_dir, replot=False)

    # # Plot eCDF
    # logger.info("\t...ecdf")
    # resulting_files_ecdf = plot_ecdf(df, output_dir=figure_dir, replot=False)

    # resulting_files = pd.concat([
    #     pd.DataFrame(resulting_files_critical_difference),
    #     pd.DataFrame(resulting_files_spearman_rank_correlation),
    #     pd.DataFrame(resulting_files_performance_per_problem),
    #     pd.DataFrame(resulting_files_boxplot_violinplot),
    #     pd.DataFrame(resulting_files_rank_over_time),
    #     pd.DataFrame(resulting_files_ecdf),
    # ]).reset_index(drop=True)
    # resulting_files.to_csv(report_dir / "resulting_files.csv", index=False)

    resulting_files = pd.read_csv(report_dir / "resulting_files.csv")
    print(resulting_files)

    order = {
        "Final Performance": [
            "critical_difference",
            "spearman_rank_correlation",
            "performance_per_problem",
            "finalperformance_boxplot",
            "finalperformance_violinplot",
        ],
        "Anytime Performance": [
            "rank_over_time",
            "ecdf",
        ],
    }

    
    for (scenario, set_id), info in resulting_files.groupby(["scenario", "set"]):
        report_tex = ""
        report_filename = report_dir / f"{report_name}_{scenario}_{set_id}.tex"
        full_report_filename = report_dir / f"full_{report_name}_{scenario}_{set_id}.tex"

        print(scenario, set_id)

        # Embed plots
        report_tex += "\\section{Plots}\n"
        for group_name, _order in order.items():
            report_tex += f"\\subsection{{{group_name}}}\n"

            for plot_type in _order:
                _info = info[info["plot_type"] == plot_type].iloc[0]
                plot_title = f"Scenario: {_info['scenario']} - Set: {_info['set']} - {_info['plot_type_pretty']}"
                plot_filename = "figures" + _info["filename"].split("figures")[-1]
                report_tex += latex_template_plot_block.replace(
                    "plot_title", plot_title
                ).replace(
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
                ).replace(
                    "explanation_text", _info["explanation"]
                )
        print(report_tex)
        break

    with open(report_filename, "w") as f:
        f.write(report_tex)

    full_report_tex = full_report_template.replace("report_tex", report_tex)
    with open(full_report_filename, "w") as f:
        f.write(full_report_tex)


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

if __name__ == "__main__":
    fire.Fire(generate_report)

