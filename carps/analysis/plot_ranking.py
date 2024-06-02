from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from autorank._util import get_sorted_rank_groups
from carps.analysis.run_autorank import calc_critical_difference, custom_latex_table, get_df_crit
from carps.analysis.utils import savefig, setup_seaborn
import seaborn as sns

def plot_ranking(gdf: pd.DataFrame, scenario: str, set_id: str, perf_col: str = "trial_value__cost_inc_norm", problem_prefix: str = "") -> None:
    fpath = Path("figures/ranking")
    fpath.mkdir(exist_ok=True, parents=True)
    identifier = f"{scenario}_{set_id}"
    label = f"tab:stat_results_{identifier}"
    result = calc_critical_difference(gdf, identifier=identifier, figsize=(8,3), perf_col=perf_col)
    create_report(result)
    table_str = custom_latex_table(result, label=label)
    fn = Path("figures/critd/" + label[len("tab:"):] + ".tex")
    fn.write_text(table_str)
    print(table_str)
    plt.show()

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse=False)
    print(sorted_ranks, names, groups)

    # DF on normalized perf values
    df_crit = get_df_crit(gdf, remove_nan=False, perf_col=perf_col)
    df_crit = df_crit.reindex(columns=names)
    df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
    df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
    plt.figure(figsize=(12, 12))
    sns.heatmap(df_crit, annot=False, fmt="g", cmap='viridis_r')
    plt.title('Performance of Optimizers per Problem (Normalized)')
    plt.ylabel('Problem ID')
    plt.xlabel('Optimizer')
    savefig(plt.gcf(), fpath / f"perf_opt_per_problem_{identifier}")
    plt.show()

    # Df on raw values
    # Optionally, plot the ranked data as a heatmap
    df_crit = get_df_crit(gdf, remove_nan=False, perf_col=perf_col)
    df_crit = df_crit.reindex(columns=names)
    df_crit.index = [i.replace(problem_prefix + "/dev/", "") for i in df_crit.index]
    df_crit.index = [i.replace(problem_prefix + "/test/", "") for i in df_crit.index]
    ranked_df = df_crit.rank(axis=1, method='min', ascending=True)

    plt.figure(figsize=(12, 12))
    sns.heatmap(ranked_df, annot=True, fmt="g", cmap='viridis_r')
    plt.title('Ranking of Optimizers per Problem')
    plt.ylabel('Problem ID')
    plt.xlabel('Optimizer')
    savefig(plt.gcf(), fpath / f"rank_opt_per_problem_{identifier}")
    plt.show()

    # Plotting the heatmap of the rank correlation matrix
    correlation_matrix = ranked_df.corr(method='spearman')
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, fmt=".2f")
    plt.title('Spearman Rank Correlation Matrix Between Optimizers')
    savefig(plt.gcf(), fpath / f"spearman_rank_corr_matrix_opt_{identifier}")
    plt.show()