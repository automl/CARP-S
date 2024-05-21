from __future__ import annotations

import fire

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from carps.analysis.process_data import load_logs
from autorank import autorank, plot_stats, create_report, latex_table
from carps.utils.loggingutils import get_logger
from carps.analysis.utils import savefig

logger = get_logger(__file__)




def calc_critical_difference(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True, identifier: str | None = None):
    perf_col: str = "trial_value__cost_inc"
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
    result = autorank(df_crit, alpha=0.05, verbose=True)
    create_report(result)

    fig, ax = plt.subplots()
    ax = plot_stats(result, ax=ax)

    if identifier is None:
        identifier = ""
    else:
        identifier = "_" + identifier        
    fn = f"figures/critd/criticaldifference{identifier}"
    savefig(fig=fig, filename=fn + ".png")
    savefig(fig=fig, filename=fn + ".pdf")

    return result


def calc(rundir: str, scenario: str = "blackbox") -> None:
    df, df_cfg = load_logs(rundir=rundir)
    calc_critical_difference(df=df[df["scenario"]==scenario], identifier=scenario)


if __name__ == "__main__":
    fire.Fire(calc)