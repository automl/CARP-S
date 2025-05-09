"""Autorank analysis for CARPS."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank._util import RankResult, get_sorted_rank_groups, rank_multiple_nonparametric, test_normality

from carps.analysis.process_data import load_logs
from carps.analysis.utils import filter_only_final_performance
from carps.utils.loggingutils import get_logger

if TYPE_CHECKING:
    from matplotlib.axis import Axis

logger = get_logger(__file__)


def custom_latex_table(
    result: RankResult, *, decimal_places: int = 3, label: str | None = None, only_tabular: bool = True
) -> str:
    """Creates a latex table from the results dataframe of the statistical analysis.

    Parameters
    ----------

    result (RankResult):
        Should be the return value the autorank function.
    decimal_places (int, default=3):
        Number of decimal places that are used for the report.
    label (str, default=None):
        Label of the table. Defaults to 'tbl:stat_results' if None.
    only_tabular (bool, default=True):
        If True, only the tabular part of the table is returned.
    """
    if label is None:
        label = "tbl:stat_results"

    table_df = result.rankdf
    columns = table_df.columns.to_list()
    if (
        result.omnibus != "bayes"
        and result.pvalue >= result.alpha
        or result.omnibus == "bayes"
        and len({"smaller", "larger"}.intersection(set(result.rankdf["decision"]))) == 0
    ):
        columns.remove("effect_size")
        columns.remove("magnitude")
    if result.posthoc == "tukeyhsd":
        columns.remove("meanrank")
    columns.insert(columns.index("ci_lower"), "CI")
    columns.remove("ci_lower")
    columns.remove("ci_upper")
    rename_map = {}
    if result.effect_size == "cohen_d":
        rename_map["effect_size"] = "$d$"
    elif result.effect_size == "cliff_delta":
        rename_map["effect_size"] = r"D-E-L-T-A"
    elif result.effect_size == "akinshin_gamma":
        rename_map["effect_size"] = r"G-A-M-M-A"
    rename_map["magnitude"] = "Magnitude"
    rename_map["mad"] = "MAD"
    rename_map["median"] = "MED"
    rename_map["meanrank"] = "MR"
    rename_map["mean"] = "M"
    rename_map["std"] = "SD"
    rename_map["decision"] = "Decision"
    format_string = "[{0[ci_lower]:." + str(decimal_places) + "f}, {0[ci_upper]:." + str(decimal_places) + "f}]"
    table_df["CI"] = table_df.agg(format_string.format, axis=1)
    table_df = table_df[columns]
    if result.omnibus == "bayes":
        table_df.loc[table_df.index[0], "decision"] = "-"
    table_df = table_df.rename(rename_map, axis="columns")

    float_format = lambda x: ("{:0." + str(decimal_places) + "f}").format(x) if not np.isnan(x) else "-"
    table_string = table_df.to_latex(float_format=float_format, na_rep="-").strip()
    table_string = table_string.replace("D-E-L-T-A", r"$\delta$")
    table_string = table_string.replace("G-A-M-M-A", r"$\gamma$")
    table_string = table_string.replace(r"p\_equal", r"$P(\textit{equal})$")
    table_string = table_string.replace(r"p\_smaller", r"$P(\textit{smaller})$")
    table_string = table_string.replace("_", r"\_")
    final_str = rf"""
\\begin{{table}}[h]
    \\caption{{Summary of populations}}
    \\label{{{label}}}
    \centering
    {table_string}
\end{{table}}
"""

    if only_tabular:
        return table_string
    return final_str


def get_df_crit(
    df: pd.DataFrame,
    budget_var: str = "n_trials_norm",
    max_fidelity: float = 1,
    soft: bool = True,  # noqa: FBT001, FBT002
    perf_col: str = "trial_value__cost_inc",
    nan_handling: str = "remove",
) -> pd.DataFrame:
    """Get the critical difference dataframe.

    1. Filter all rundata for the final performance (objective function value of incumbent at the end of the
        optimization run).
    2. Calculate the mean performance for the different seeds per optimizer and task.
    3. Pivot the table such that we have 2D table with tasks as rows and optimizers as columns.
    4. Handle nans: Remove rows with nans, replace nans by highest (worst) value, or keep them.

    Args:
        df (pd.DataFrame): The dataframe.
        budget_var (str, optional): The budget variable. Defaults to "n_trials_norm".
        max_fidelity (float, optional): The maximum budget. Defaults to 1.
        soft (bool, optional): Whether to use a soft filter: If no entry at max_fidelity is available,
            use the performance at the last trial. Defaults to True.
        perf_col (str, optional): The performance column. Defaults to "trial_value__cost_inc".
        nan_handling (str, optional): How to handle nans. Can be "remove", "keep", "replace_by_highest".
            Defaults to "remove".

    Returns:
        pd.DataFrame: The critical difference dataframe.
    """
    df = filter_only_final_performance(df=df, budget_var=budget_var, max_fidelity=max_fidelity, soft=soft)  # noqa: PD901

    # Work on mean of different seeds
    df_crit = df.groupby(["optimizer_id", "task_id"])[perf_col].apply(np.nanmean).reset_index()

    df_crit = df_crit.pivot_table(index="task_id", columns="optimizer_id", values=perf_col)

    if nan_handling == "remove":
        nan_ids = np.array([np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])
        lost = df_crit[nan_ids]

        # Rows are tasks, cols are optimizers
        if len(lost) > 0:
            df_crit = df_crit[~nan_ids]
            logger.info(f"Lost following experiments: {lost}")
    elif nan_handling == "keep":
        pass
    elif nan_handling == "replace_by_highest":
        nan_ids = np.array([np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])
        max_val = df_crit[perf_col].max()
        df_crit.loc[nan_ids, perf_col] = max_val
        logger.info(f"Replaced nans by max val {max_val}: {df_crit}")
    else:
        raise ValueError(f"Unknown nan handling {nan_handling}. Can only do `remove`, `keep`, `replace_by_highest`.")

    return df_crit


def calc_critical_difference(
    df: pd.DataFrame,
    identifier: str | None = None,
    figsize: tuple[int | float, int | float] = (12, 8),
    perf_col: str = "trial_value__cost_inc_norm",
    plot_diagram: bool = True,  # noqa: FBT001, FBT002,
    calc_df_crit: bool = True,  # noqa: FBT001, FBT002
) -> RankResult:
    """Calculate the critical difference.

    Args:
        df (pd.DataFrame): The dataframe.
        identifier (str, optional): Identifier for the plot. Defaults to None.
        figsize (tuple[int | float, int | float], optional): Figure size. Defaults to (12, 8).
        perf_col (str, optional): The performance column. Defaults to "trial_value__cost_inc_norm".
        plot_diagram (bool, optional): Whether to plot the diagram. Defaults to True.
        calc_df_crit (bool, optional): Whether to calculate the df_crit (averaging over seeds). Defaults to True. If
            False, df is used as is.

    Returns:
        RankResult: The rank result.
    """
    df_crit = get_df_crit(df, perf_col=perf_col) if calc_df_crit else df

    # result = autorank(df_crit, alpha=0.05, verbose=True)
    # create_report(result)

    # fig, ax = plt.subplots(figsize=(6,2))
    # ax = plot_stats(result, ax=ax)

    # if identifier is None:
    #     identifier = ""
    # else:
    #     identifier = "_" + identifier
    # fn = f"figures/critd/criticaldifference{identifier}"
    # savefig(fig=fig, filename=fn + ".png")
    # savefig(fig=fig, filename=fn + ".pdf")

    return cd_evaluation(
        df_crit,
        maximize_metric=False,
        ignore_non_significance=True,
        output_path=f"figures/critd/cd{identifier}",
        figsize=figsize,
        plot_diagram=plot_diagram,
    )


def calc(rundir: str, task_type: str = "blackbox") -> None:
    """Calculate the critical difference for a specific task_type.

    Args:
        rundir (str): The run directory.
        task_type (str, optional): The task_type. Defaults to "blackbox".
    """
    df, df_cfg = load_logs(rundir=rundir)
    calc_critical_difference(df=df[df["task_type"] == task_type], identifier=task_type)


"""Code for CD Plots
Requirements: the usual (pandas, matplotlib, numpy) and autorank==1.1.3

Source: https://gist.github.com/LennartPurucker/cf4616512529e29c123608b6c2c4a7e9
"""


def _custom_cd_diagram(result: RankResult, reverse: bool, ax: Axis, width: float) -> Axis:  # noqa: C901, FBT001, PLR0915
    """!TAKEN FROM AUTORANK WITH MODIFICATIONS!"""

    def plot_line(line: list[tuple[float, float]], color: str = "k", **kwargs: Any) -> None:
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x: float, y: float, s: str, *args: tuple, **kwargs: Any) -> None:
        ax.text(x / width, y / height, s, *args, **kwargs)

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse)
    cd = result.cd

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank: float) -> float:
        relative_rank = rank - lowv if not reverse else highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.1
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).3
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick: int | float | None = None
    for a in [*list(np.arange(lowv, highv, 0.5)), highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv + 1):
        assert tick is not None, "Cannot plot, tick was not set."
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line(
            [(rankpos(sorted_ranks.iloc[i]), cline), (rankpos(sorted_ranks.iloc[i]), chei), (textspace - 0.1, chei)],
            linewidth=0.7,
        )
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line(
            [
                (rankpos(sorted_ranks.iloc[i]), cline),
                (rankpos(sorted_ranks.iloc[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=0.7,
        )
        plot_text(textspace + scalewidth + 0.2, chei, names[i], ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)
    distanceh += 0.15
    bigtick /= 2
    plot_line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
    plot_line([(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)], linewidth=0.7)
    plot_line([(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)], linewidth=0.7)
    plot_text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:  # noqa: E741
        plot_line([(rankpos(sorted_ranks[l]) - side, start), (rankpos(sorted_ranks[r]) + side, start)], linewidth=2.5)
        start += no_sig_height

    return ax


def cd_evaluation(
    performance_per_dataset: pd.DataFrame,
    maximize_metric: bool,  # noqa: FBT001
    output_path: str | None = None,
    ignore_non_significance: bool = False,  # noqa: FBT001, FBT002
    plt_title: str | None = None,
    figsize: tuple[int, int] | tuple[float, float] = (12, 8),
    plot_diagram: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> RankResult:
    """Run critical difference evaluation.

    Performance per dataset is  a dataframe that stores the performance (with respect to a metric) for  set of
    configurations / models / algorithms per dataset. In  detail, the columns are individual configurations.
    rows are datasets and a cell is the performance of the configuration for  dataset.

    Args:
        performance_per_dataset (pd.DataFrame): Performance per dataset.
        maximize_metric (bool): Whether to maximize the metric.
        output_path (str, optional): Path to save the figure. Defaults to None.
        ignore_non_significance (bool, optional): Whether to ignore non-significance. Defaults to False.
        plt_title (str, optional): Title of the plot. Defaults to None.
        figsize (tuple[int | float], optional): Figure size. Defaults to (12, 8).
        plot_diagram (bool, optional): Whether to plot the diagram. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        RankResult: Rank result.
    """
    # -- Preprocess data for autorank
    rank_data = performance_per_dataset.copy() * -1 if maximize_metric else performance_per_dataset.copy()
    rank_data = rank_data.reset_index(drop=True)
    rank_data = pd.DataFrame(rank_data.values, columns=list(rank_data))

    # -- Settings for autorank
    alpha = 0.05
    effect_size = None
    order = "ascending"  # always due to the preprocessing
    alpha_normality = alpha / len(rank_data.columns)
    all_normal, pvals_shapiro = test_normality(rank_data, alpha_normality, verbose)

    # -- Friedman-Nemenyi
    res = rank_multiple_nonparametric(rank_data, alpha, verbose, all_normal, order, effect_size, None)

    result = RankResult(
        rankdf=res.rankdf,
        pvalue=res.pvalue,
        cd=res.cd,
        omnibus=res.omnibus,
        posthoc=res.posthoc,
        all_normal=all_normal,
        pvals_shapiro=pvals_shapiro,
        homoscedastic=None,
        pval_homogeneity=None,
        homogeneity_test=None,
        alpha=alpha,
        alpha_normality=alpha_normality,
        num_samples=len(rank_data),
        posterior_matrix=None,
        decision_matrix=None,
        rope=None,
        rope_mode=None,
        effect_size=res.effect_size,
        force_mode=None,
    )
    is_significant = True
    if result.pvalue >= result.alpha:
        if ignore_non_significance:
            warnings.warn("Result is not significant and results of the plot may be misleading!", stacklevel=2)
            is_significant = False
        else:
            raise ValueError(
                "Result is not significant and results of the plot may be misleading. If you still want to see the "
                "CD plot, set ignore_non_significance to True."
            )

    # -- Plot
    if plot_diagram:
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({"font.size": 16})
        _custom_cd_diagram(result=result, reverse=False, ax=ax, width=figsize[0])  # order == "descending", ax, 8)
        if plt_title or not is_significant:
            plt_title = ""
            if not is_significant:
                plt_title += "(non-significant)"
            plt.title(plt_title)
        plt.tight_layout()
        if output_path:
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path + ".png", transparent=True, bbox_inches="tight")
            plt.savefig(output_path + ".pdf", transparent=True, bbox_inches="tight")

        plt.show()
        plt.close()

    return result


if __name__ == "__main__":
    fire.Fire(calc)
