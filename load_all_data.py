from __future__ import annotations

# from carps.analysis.process_data import get_interpolated_performance_df, load_logs, process_logs
import time

import pandas as pd
from carps.analysis.gather_data import (
    convert_mixed_types_to_str,
    load_set,
    maybe_convert_cost_dtype,
)

paths = {
    # "BBfull": {
    #     "full": [
    #     "runs/SMAC3-BlackBoxFacade",
    #     "runs/RandomSearch",
    #     "runs/Nevergrad-CMA-ES",
    # ]},
    # "MOfull": {
    #     "full": ["runs_MO"]
    # },
    "BBsubset": {
        "dev": ["runs_subset_BB/dev"],
        "test": ["runs_subset_BB/test"],
    },
    "MFsubset": {
        "dev": ["runs_subset_MF/dev"],
        "test": ["runs_subset_MF/test"],
    },
    "MOsubset": {
        "dev": ["runs_subset_MO/dev"],
        "test": ["runs_subset_MO/test"],
    },
    "MOMFsubset": {
        "dev": ["runs_subset_MOMF/dev"],
        "test": ["runs_subset_MOMF/test"],
    },
}
# subset = "BBsubset"
# task_prefix = "blackbox/20"

# subset = "MFsubset"
# task_prefix = "multifidelity/20"

# subset = "MOsubset"
# task_prefix = "multiobjective/10"

# subset = "MOMFsubset"
# task_prefix = "momf/9"

def fix_floats(key: str, df: pd.DataFrame) -> pd.DataFrame:
    df[key] = df[key].apply(maybe_convert_cost_dtype)
    return df

all_df = []
all_df_cfg = []
for subset, ps in paths.items():
    start = time.time()
    print("Load subset", subset)
    loaded = [load_set(paths=ps, set_id=set_id) for set_id, ps in paths[subset].items()]

    df = pd.concat([d for d, _ in loaded]).reset_index(drop=True)
    df_cfg = pd.concat([d for _, d in loaded]).reset_index(drop=True)
    all_df.append(df)
    all_df_cfg.append(df_cfg)
    end = time.time()
    print(f"Loaded {len(df)} logs in {end - start:.2f} s")
all_df = pd.concat(all_df).reset_index(drop=True)
all_df_cfg = pd.concat(all_df_cfg).reset_index(drop=True)

# all_df = fix_floats("trial_info__budget", all_df)
# all_df = fix_floats("trial_info__normalized_budget", all_df)
all_df = convert_mixed_types_to_str(all_df)

all_df.to_parquet("logs_combined.parquet")
all_df_cfg.to_parquet("logs_combined_cfg.parquet")