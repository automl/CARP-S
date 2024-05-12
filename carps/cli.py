"""
# CLI.

This module defines command-line options using flags.

This includes the entry point for the programs execution.
"""

import subprocess
from typing import Any

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "create_cluster_configs",
    short_name="c",
    default=False,
    help="Create cluster configs in the database. Passing paths to the "
    "optimizer and problem config to be used is required.",
)
flags.DEFINE_boolean(
    "run_from_db",
    short_name="db",
    default=False,
    help="Start runs for all configs in the database.",
)
flags.DEFINE_boolean(
    "run",
    short_name="r",
    default=False,
    help="Start run for provided optimizer, problem, and seed. Passing paths to "
    "the optimizer and problem config to be used is required, as well as "
    "providing a seed.",
)
flags.DEFINE_boolean(
    "check_missing", short_name="cm", default=False, help="Check for missing runs."
)
flags.DEFINE_boolean(
    "gather_data",
    short_name="gd",
    default=False,
    help="Gather logs from the run files.",
)
flags.DEFINE_string(
    "optimizer", short_name="o", default=None, help="Path to optimizer config."
)
flags.DEFINE_string(
    "problem", short_name="p", default=None, help="Path to problem config."
)
flags.DEFINE_string(
    "seed",
    short_name="s",
    default=None,
    help="Seed to be used when running the experiment.",
)
flags.DEFINE_string(
    "rundir",
    short_name="rd",
    default="runs",
    help="Path to the run directory where results and logs are stored.",
)


def main(argv: Any) -> None:
    """Call the function to execute."""
    if FLAGS.create_cluster_configs:
        if FLAGS.optimizer is None or FLAGS.problem is None:
            print("Please provide optimizer and problem.")
            return
        subprocess.call(
            [
                "python",
                "-m",
                "carps.container.create_cluster_configs",
                f"+optimizer={FLAGS.optimizer}",
                f"+problem={FLAGS.problem}",
            ]
        )
    if FLAGS.run_from_db:
        subprocess.call(["python", "-m", "carps.run_from_db"])
    if FLAGS.run:
        if FLAGS.optimizer is None or FLAGS.problem is None or FLAGS.seed is None:
            print("Please provide optimizer, problem and seed.")
            return
        subprocess.call(
            [
                "python",
                "-m",
                "carps.run",
                f"+optimizer={FLAGS.optimizer}",
                f"+problem={FLAGS.problem}",
                f"seed={FLAGS.seed}",
            ]
        )
    if FLAGS.check_missing:
        subprocess.call(["python", "-m", "carps.utils.check_missing", FLAGS.rundir])
    if FLAGS.gather_data:
        subprocess.call(["python", "-m", "carps.analysis.gather_data", FLAGS.rundir])


if __name__ == "__main__":
    pass

app.run(main)
