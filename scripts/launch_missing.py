from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import fire

sbatch_template="""#!/bin/bash
#for configuration options see: https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/12944324/Running+Compute+Jobs
#SBATCH -N 1
#SBATCH -n 4                                    # Number of CPUs you want on that machine (<=128)
#SBATCH --mem 8GB                               # Amount of RAM you want (<=239GB), e.g. 64GB
#SBATCH -J adoe                                 # Name of your job - adjust as you like
#SBATCH -A hpc-prf-intexml                      # Project name, do not change
#SBATCH -t 02:00:00                             # Timelimit of the job. See documentation for format.
#SBATCH --mail-type fail                        # Send an email, if the job fails.
#SBATCH -p normal                               # Normal job, no GPUs
#SBATCH -e slurmout/slurm-%A_%a.out

{command}
"""

def launch(debug: bool = False, extra_config: str = ""):
    fn_cmd = Path("scripts/run_missing.sh")
    if not fn_cmd.is_file():
        msg = f"No missing runs at {fn_cmd}. Missing runs are generated via plot.ipynb -> plot_interval_estimates -> get_final_performance_dict."
        raise RuntimeError(msg)

    path_sbatch = Path("scripts/missing")
    if path_sbatch.exists():
        shutil.rmtree(path_sbatch)
    path_sbatch.mkdir(exist_ok=True, parents=True)


    with open(fn_cmd) as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        cmd = line
        index = len("python -m carps.run")
        cmd = cmd[:index] + extra_config + cmd[index+1:]
        print(cmd)

        filecontent = sbatch_template.format(command=cmd)
        fn = path_sbatch / f"run_{i}.sh"
        with open(fn, "w") as file:
            file.write(filecontent)

        if not debug:
            fire_cmd = f"sbatch {fn}"
            subprocess.Popen(fire_cmd, shell=True)

if __name__ == "__main__":
    fire.Fire(launch)
