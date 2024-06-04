from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

# Rliable
from rliable import (
    library as rly,
    metrics,
    plot_utils,
)

from carps.analysis.utils import savefig
from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.overridefinder import find_override

setup_logging()
logger = get_logger(__file__)


def get_final_performance_dict(
    performance_data: pd.DataFrame,
    key_method: str,
    key_performance: str,
    key_instance: str,
    budget_var: str = "n_trials_norm",
    max_budget: float = 1,
) -> dict[str, Any]:
    """Generate performance dict for rliable.

    Parameters
    ----------
    D : pd.DataFrame
        Dataframe with final performance
    key_method : str
        Key of the method, e.g. optimizer_id
    key_performance : str
        Key of the performance column, e.g. cost_inc
    key_instance : str
        Key of the problem instancde, e.g. problem_id
    budget_var : str, optional
        The budget name, by default `n_trials_norm`.
    max_budget : float, optional
        The maximum budget considered. 1 for normalized budget vars.

    Returns:
    --------
    dict[str, Any]
        Dict with method as keys and performance values with shape [n seeds x m instances]
    """
    performance_data = performance_data[np.isclose(performance_data[budget_var], max_budget)]
    perf_dict = {}
    dropped: list = []
    seeds = set(performance_data["seed"].unique())
    instances = set(performance_data[key_instance].unique())
    combos = set(itertools.product(instances, seeds))
    override_commands: list[str] = []
    for gid, gdf in performance_data.groupby(key_method):
        gdf = gdf.reset_index()
        n_seeds = gdf["seed"].nunique()
        n_instances = gdf[key_instance].nunique()
        P = gdf[key_performance].to_numpy()

        _combos = gdf.loc[:, ["problem_id", "seed"]].to_numpy()
        _combos = set(map(tuple, _combos))

        missing = list(combos.difference(_combos))

        if len(missing) > 0:
            dropped.append(
                {
                    "method": gid,
                    "missing": missing,
                }
            )
            optimizer_override = find_override(optimizer_id=gid) or f"+optimizer={gid}"

            for _missing in missing:
                problem_override = find_override(problem_id=_missing[0]) or f"+problem={_missing[0]}"
                seed_override = f"seed={_missing[1]}"
                override_cmd = f"{seed_override} {problem_override} {optimizer_override}"
                override_commands.append(override_cmd)
        else:
            # if len(P) == n_seeds * n_instances:
            P = P.reshape((n_seeds, n_instances))
            perf_dict[gid] = P
    if len(dropped) > 0:
        logger.info("Dropped following incomplete methods:")
        logger.info(dropped)

        base_command = "python -m carps.run {override}\n"
        fn = Path(__file__).parent.parent.parent / "scripts/missing/run_missing.sh"
        fn.parent.mkdir(exist_ok=True, parents=True)
        with open(fn, "w") as file:
            for override in override_commands:
                file.write(base_command.format(override=override))

        logger.info(f"Created override commands at {fn}")
    return perf_dict


def calculate_interval_estimates(
    final_performance_dict: pd.DataFrame, metrics: dict[str, callable], repetitions: int = 5000
):
    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.
    def aggregate_func(x):
        return np.array([agg_fun(x) for agg_fun in metrics.values()])

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        final_performance_dict, aggregate_func, reps=repetitions
    )

    with open("final_performance_dict.pickle", "wb") as handle:
        pickle.dump(final_performance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("aggregate_scores.pickle", "wb") as handle:
        pickle.dump(aggregate_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("aggregate_score_cis.pickle", "wb") as handle:
        pickle.dump(aggregate_score_cis, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(final_performance_dict)
    print(aggregate_scores)

    return aggregate_scores, aggregate_score_cis


def plot_interval_estimates(
    performance_data: pd.DataFrame,
    load_from_pickle: bool = False,
    figure_filename: str = "figures/plot_interval_estimates.pdf",
):
    _metrics = {
        "IQM": metrics.aggregate_iqm,
        "mean": metrics.aggregate_mean,
        "median": metrics.aggregate_median,
    }
    metric_names = list(_metrics.keys())

    if not load_from_pickle:
        final_performance_dict = get_final_performance_dict(
            performance_data=performance_data,
            key_method="optimizer_id",
            key_performance="trial_value__cost",
            key_instance="problem_id",
        )
        aggregate_scores, aggregate_score_cis = calculate_interval_estimates(
            final_performance_dict=final_performance_dict, metrics=_metrics, repetitions=5000
        )
    else:
        final_performance_dict = pickle.load("final_performance_dict.pickle")
        aggregate_scores = pickle.load("aggregate_scores.pickle")
        aggregate_score_cis = pickle.load("aggregate_score_cis.pickle")

    algorithms = list(final_performance_dict.keys())

    sns.set_style("white")

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=metric_names,
        algorithms=algorithms,
        xlabel="Cost",
        xlabel_y_coordinate=-0.6,
    )
    savefig(fig, figure_filename)

    return fig, axes

    # fig, axes = plot_utils.plot_interval_estimates(
    #     aggregate_scores, aggregate_score_cis,
    #     metric_names=metric_names,
    #     algorithms=algorithms, xlabel="Performance",
    #     xlabel_y_coordinate=0.025
    # )
    # fig.savefig("plot_rliable_all.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    perf = pd.read_csv("/home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/perf.csv")
    plot_interval_estimates(performance_data=perf)
