from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns
from autorank import create_report
from autorank._util import get_sorted_rank_groups

from carps.analysis.run_autorank import calc_critical_difference, custom_latex_table, get_df_crit
from carps.analysis.utils import savefig, get_color_palette, filter_only_final_performance

if TYPE_CHECKING:
    import pandas as pd


def plot_ranking(
    gdf: pd.DataFrame,
    scenario: str,
    set_id: str,
    perf_col: str = "trial_value__cost_inc_norm",
    problem_prefix: str = "",
) -> None:
    fpath = Path("figures/ranking")
    fpath.mkdir(exist_ok=True, parents=True)
    identifier = f"{scenario}_{set_id}"
    label = f"tab:stat_results_{identifier}"
    result = calc_critical_difference(gdf, identifier=identifier, figsize=(8, 3), perf_col=perf_col)
    # print(result)
    # try: 
    #     create_report(result)
    # except Exception as e:
    #     print(e)
    # table_str = custom_latex_table(result, label=label)
    # fn = Path("figures/critd/" + label[len("tab:") :] + ".tex")
    # fn.write_text(table_str)
    # print(table_str)
    # plt.show()

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)
    # print(sorted_ranks, names, groups)

    # # DF on normalized perf values
    # df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
    # df_crit = df_crit.reindex(columns=names)
    # df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
    # df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(df_crit, annot=False, fmt="g", cmap="viridis_r")
    # plt.title("Performance of Optimizers per Problem (Normalized)")
    # plt.ylabel("Problem ID")
    # plt.xlabel("Optimizer")
    # savefig(plt.gcf(), fpath / f"perf_opt_per_problem_{identifier}")
    # plt.show()

    # # Df on raw values
    # # Optionally, plot the ranked data as a heatmap
    # df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
    # df_crit = df_crit.reindex(columns=names)
    # df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
    # df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
    # ranked_df = df_crit.rank(axis=1, method="min", ascending=True)

    # plt.figure(figsize=(12, 12))
    # sns.heatmap(ranked_df, annot=True, fmt="g", cmap="viridis_r")
    # plt.title("Ranking of Optimizers per Problem")
    # plt.ylabel("Problem ID")
    # plt.xlabel("Optimizer")
    # savefig(plt.gcf(), fpath / f"rank_opt_per_problem_{identifier}")
    # plt.show()

    # # Plotting the heatmap of the rank correlation matrix
    # correlation_matrix = ranked_df.corr(method="spearman")
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f")
    # plt.title("Spearman Rank Correlation Matrix Between Optimizers")
    # savefig(plt.gcf(), fpath / f"spearman_rank_corr_matrix_opt_{identifier}")
    # plt.show()


    # combined
    ncols = 2
    nrows = 3
    right = 0.6
    h = 0.9
    w = 1
    factor = 30
    hspace = 0.3
    wspace = 0.7

    fig = plt.figure(layout=None, facecolor="white", figsize=(w * factor, h * factor))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, left=0.05, right=right,
                        hspace=hspace, wspace=wspace
                        )

    # Perf per problem (normalized)
    ax0 = fig.add_subplot(gs[:, :-1])
    df_crit = get_df_crit(gdf, nan_handling="keep", perf_col=perf_col)
    df_crit = df_crit.reindex(columns=names)
    df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
    df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
    ax0 = sns.heatmap(df_crit, annot=False, fmt="g", cmap="viridis_r", ax=ax0, cbar_kws={"shrink": 0.8, "aspect": 30})
    ax0.set_title("Final Performance per Problem (Normalized)")
    ax0.set_ylabel("Problem ID")
    ax0.set_xlabel("Optimizer")


    df_finalperf = filter_only_final_performance(df=gdf)
    sorter = names
    df_finalperf = df_finalperf.sort_values(by="optimizer_id", key=lambda column: column.map(lambda e: sorter.index(e)))
    palette = get_color_palette(df=gdf)
    ax1 = fig.add_subplot(gs[0, -1])
    x="n_trials_norm"
    y="trial_value__cost_inc_norm"
    x = y
    hue="optimizer_id"
    y = hue
    ax1 = sns.boxplot(
        data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax1
    )
    ax1.set_title("Final Performance (Normalized)")

    ax2 = fig.add_subplot(gs[1, -1])
    ax2 = sns.violinplot(
        data=df_finalperf, y=y, x=x, hue=hue, palette=palette, ax=ax2, cut=0
    )
    ax2.set_title("Final Performance (Normalized)")

    # Spearman rank correlation
    ax3 = fig.add_subplot(gs[2, -1])
    ranked_df = df_crit.rank(axis=1, method="min", ascending=True)
    correlation_matrix = ranked_df.corr(method="spearman")
    ax3 = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f", ax=ax3)
    ax3.set_title("Spearman Rank Correlation Matrix\nBetween Optimizers")

    # fig.set_tight_layout(True)

    savefig(fig, fpath / f"final_per_combined_{identifier}")

    plt.show()
