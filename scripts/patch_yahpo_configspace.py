from __future__ import annotations

from pathlib import Path
import json

print("Fixing yahpo configspace files.")
path = Path("smacbenchmarking/benchmark_data/yahpo_data")
configspace_paths = list(path.glob("**/config_space.json"))
for p in configspace_paths:
    print("\t", p)
    with open(p, "r") as file:
        cs = json.load(file)
    hps = cs["hyperparameters"]
    new_hps = []
    for hp in hps:
        if "q" not in hp:
            hp["q"] = None
        new_hps.append(hp)
    cs["hyperparameters"] = hps
    with open(p, "w") as file:
        json.dump(cs, file, indent="\t")

print("Done!")
