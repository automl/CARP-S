from __future__ import annotations

from carps.analysis.gather_data import load_set, convert_mixed_types_to_str
from carps.analysis.utils import filter_only_final_performance
import pandas as pd
import multiprocessing

paths = {
    "BBsubset": {
        "dev": ["../runs_subset_BB/dev"],
        "test": ["../runs_subset_BB/test"],
    },
    "MFsubset": {
        "dev": ["../runs_subset_MF/dev"],
        "test": ["../runs_subset_MF/test"],
    },
    "MOsubset": {
        "dev": ["../runs_subset_MO/dev"],
        "test": ["../runs_subset_MO/test"],
    },
    "MOMFsubset": {
        "dev": ["../runs_subset_MOMF/dev"],
        "test": ["../runs_subset_MOMF/test"],
    },
}



# args = []
# for item in paths.values():
#     for k,v in item.items():
#         args.append((v,k))
# res = [load_set(paths=a[0], set_id=a[1]) for a in args]
# df = pd.concat([r[0] for r in res]).reset_index(drop=True)
# df = convert_mixed_types_to_str(df)
# df.to_parquet("rundata.parquet")

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







