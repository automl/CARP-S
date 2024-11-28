from __future__ import annotations

from pathlib import Path
import pandas as pd
import subprocess

rundir = "run-data-MO"
ks = [10,20]


rundir = Path(rundir)
fullset_fn = rundir / "df_crit.csv"

print(fullset_fn)

fullset = pd.read_csv(fullset_fn, index_col="problem_id")

points = fullset.values
n_points, dimension = points.shape

pointfile ="pointfile.txt"

fullset.to_csv(pointfile, sep=",", index=False, header=False)

# SHIFT_TRIES=${SHIFT_TRIES:-5000} ${executable} ${input_file} ${dimension} ${num_points} ${value} subset_${value}.txt"

n_reps = 5000
executable = "./a.out"


for k in ks:
    outfile = f"subset_{k}.txt"
    command = f"export SHIFT_TRIES={n_reps}; {executable} {pointfile} {dimension} {n_points} {k} {outfile}"
    result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    print(result)
    break

