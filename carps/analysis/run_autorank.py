from __future__ import annotations

import fire

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from carps.analysis.process_data import load_logs
from autorank import autorank, plot_stats, create_report
from carps.utils.loggingutils import get_logger
from carps.analysis.utils import savefig

logger = get_logger(__file__)

def get_df_crit(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True, perf_col: str = "trial_value__cost_inc") -> pd.DataFrame:
    if not soft:
        df = df[np.isclose(df[budget_var], max_budget)]
    else:
        df = df[df.groupby(["optimizer_id", "problem_id", "seed"])[budget_var].transform(lambda x: x == x.max())]
    
    # Work on mean of different seeds
    df_crit = df.groupby(["optimizer_id", "problem_id"])[perf_col].apply(np.nanmean).reset_index()
       
    df_crit = df_crit.pivot(
        index="problem_id",
        columns="optimizer_id",
        values=perf_col
    )
    
    lost = df_crit[np.array([np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])] 

    # Rows are problems, cols are optimizers
    df_crit = df_crit[np.array([not np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])]     
    logger.info(f"Lost following experiments: {lost}")

    return df_crit

def calc_critical_difference(df: pd.DataFrame, identifier: str | None = None):
    df_crit = get_df_crit(df)

    result = autorank(df_crit, alpha=0.05, verbose=True)
    create_report(result)

    fig, ax = plt.subplots(figsize=(6,2))
    ax = plot_stats(result, ax=ax)

    if identifier is None:
        identifier = ""
    else:
        identifier = "_" + identifier        
    fn = f"figures/critd/criticaldifference{identifier}"
    savefig(fig=fig, filename=fn + ".png")
    savefig(fig=fig, filename=fn + ".pdf")


    cd_evaluation(
        df_crit, 
        maximize_metric=False, 
        ignore_non_significance=False, 
        output_path=f"figures/critd/cd{identifier}"
    )

    return result


def calc(rundir: str, scenario: str = "blackbox") -> None:
    df, df_cfg = load_logs(rundir=rundir)
    calc_critical_difference(df=df[df["scenario"]==scenario], identifier=scenario)



"""Code for CD Plots
Requirements: the usual (pandas, matplotlib, numpy) and autorank==1.1.3

Source: https://gist.github.com/LennartPurucker/cf4616512529e29c123608b6c2c4a7e9
"""

import math
import warnings

from autorank._util import RankResult, get_sorted_rank_groups, rank_multiple_nonparametric, test_normality


def _custom_cd_diagram(result, reverse, ax, width):
    """
    !TAKEN FROM AUTORANK WITH MODIFICATIONS!
    """

    def plot_line(line, color="k", **kwargs):
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse)
    cd = result.cd

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
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

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline), (rankpos(sorted_ranks[i]), chei), (textspace - 0.1, chei)], linewidth=0.7)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline), (rankpos(sorted_ranks[i]), chei), (textspace + scalewidth + 0.1, chei)], linewidth=0.7)
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
    for l, r in groups:
        plot_line([(rankpos(sorted_ranks[l]) - side, start), (rankpos(sorted_ranks[r]) + side, start)], linewidth=2.5)
        start += no_sig_height

    return ax


def cd_evaluation(performance_per_dataset, maximize_metric, output_path=None, ignore_non_significance=False, plt_title=None):
    """Performance per dataset is  a dataframe that stores the performance (with respect to a metric) for  set of
    configurations / models / algorithms per dataset. In  detail, the columns are individual configurations.
    rows are datasets and a cell is the performance of the configuration for  dataset.
    """

    # -- Preprocess data for autorank
    if maximize_metric:
        rank_data = performance_per_dataset.copy() * -1
    else:
        rank_data = performance_per_dataset.copy()
    rank_data = rank_data.reset_index(drop=True)
    rank_data = pd.DataFrame(rank_data.values, columns=list(rank_data))

    # -- Settings for autorank
    alpha = 0.05
    effect_size = None
    verbose = True
    order = "ascending"  # always due to the preprocessing
    alpha_normality = alpha / len(rank_data.columns)
    all_normal, pvals_shapiro = test_normality(rank_data, alpha_normality, verbose)

    # -- Friedman-Nemenyi
    res = rank_multiple_nonparametric(rank_data, alpha, verbose, all_normal, order, effect_size, None)

    result = RankResult(
        res.rankdf,
        res.pvalue,
        res.cd,
        res.omnibus,
        res.posthoc,
        all_normal,
        pvals_shapiro,
        None,
        None,
        None,
        alpha,
        alpha_normality,
        len(rank_data),
        None,
        None,
        None,
        None,
        res.effect_size,
        None,
    )
    print(res.rankdf)
    if result.pvalue >= result.alpha:
        if ignore_non_significance:
            warnings.warn("Result is not significant and results of the plot may be misleading!")
        else:
            raise ValueError("Result is not significant and results of the plot may be misleading. If you still want to see the CD plot, set" +
                             " ignore_non_significance to True.")

    # -- Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update({"font.size": 16})
    _custom_cd_diagram(result, order == "ascending", ax, 8)
    if plt_title:
        plt.title(plt_title)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path + ".png", transparent=True, bbox_inches="tight")
        plt.savefig(output_path + ".pdf", transparent=True, bbox_inches="tight")
        
    plt.show()
    plt.close()

    return result


# if __name__ == "__main__":
#     input_data = pd.read_csv("./benchmark_binary_1hour.csv")  # assume table format where each row represents some experiment run

#     # Pivot for desired metric to create the performance per dataset table
#     performance_per_dataset = input_data.pivot(index="task", columns="framework", values="result")

#     # shows the plot by default
#     cd_evaluation(performance_per_dataset, maximize_metric=True, output_path=None, ignore_non_significance=False)

if __name__ == "__main__":
    fire.Fire(calc)