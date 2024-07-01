from __future__ import annotations
import pandas as pd
from carps.analysis.gather_data import load_set, convert_mixed_types_to_str

def concat_rundata():
    paths = {
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

    args = []
    for item in paths.values():
        for k,v in item.items():
            from pathlib import Path
            args.append((v,k))
    res = [load_set(paths=a[0], set_id=a[1]) for a in args]
    df = pd.concat([r[0] for r in res]).reset_index(drop=True)
    df = convert_mixed_types_to_str(df)
    df.to_parquet("rundata.parquet")

if __name__ == "__main__":
    concat_rundata()