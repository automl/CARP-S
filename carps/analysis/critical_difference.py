# Critical Differences
from critdd import Diagram
import pandas as pd
import numpy as np
import fire

from carps.utils.loggingutils import get_logger
from carps.analysis.process_data import load_logs

logger = get_logger(__file__)

def calc_critical_difference(df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True):
    perf_col: str = "trial_value__cost_inc"
    if not soft:
        df = df[np.isclose(df[budget_var], max_budget)]
    else:
        df = df[df.groupby(["optimizer_id", "problem_id", "seed"])[budget_var].transform(lambda x: x == x.max())]
    df_crit = df.groupby(["optimizer_id", "problem_id"])[perf_col].apply(np.nanmean).reset_index()
       
    df_crit = df_crit.pivot(
        index="problem_id",
        columns="optimizer_id",
        values=perf_col
    )
    
    lost = df_crit[np.array([np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])] 
    df_crit = df_crit[np.array([not np.any(np.isnan(d.values)) for _, d in df_crit.iterrows()])]     
    logger.info(f"Lost following experiments: {lost}")
    diagram = Diagram(
        df_crit.to_numpy(),
        treatment_names = df_crit.columns,
        maximize_outcome = False
    )
    summary = f"""
Methods: {list(df_crit.columns)}
Average ranks: {diagram.average_ranks}
Groups: {diagram.get_groups(alpha=.05, adjustment='holm')}
"""
    logger.info(summary)

    # export the diagram to a file
    diagram.to_file(
        "criticaldifference.tex",
        alpha = .05,
        adjustment = "holm",
        reverse_x = True,
        axis_options = {"title": "Critical Difference"},

    )

def calc(rundir: str) -> None:
    df, df_cfg = load_logs(rundir=rundir)
    calc_critical_difference(df=df)


if __name__ == "__main__":
    fire.Fire(calc)
