from __future__ import annotations

from carps.analysis.utils import filter_only_final_performance
import pandas as pd
import multiprocessing

concat_rundata()

df  = pd.read_parquet("rundata.parquet")
df_final = filter_only_final_performance(df=df, budget_var="n_trials")

def calc(x: pd.DataFrame) -> float:
    n_optimizers = x["optimizer_id"].nunique()
    t = x["time"].sum() / n_optimizers
    return t / 3600

runtime_df = df_final.groupby(by=["scenario"]).apply(calc)
runtime_df.name = "time"
print(runtime_df)
runtime_df.to_csv("runtimes.csv")


df_rt = pd.read_csv("runtimes.csv", index_col="scenario").map(int)
total = df_rt.sum()

total = pd.DataFrame(total).T
total.index = ['total']

df_rt = pd.concat([df_rt, total], axis=0)

latex_str = df_rt.to_latex(
    buf="runtimes.tex",
    caption="Runtimes in CPU Hours per Scenario",
    label="tab:runtimes",
    index_names=True
)







