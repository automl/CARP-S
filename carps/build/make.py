"""This module is used to run make commands from the root of the project.

Usage: python -m carps.build.make <target1> <target2> ...
E.g.: python -m carps.build.make benchmark_bbob optimizer_smac
"""
from __future__ import annotations
import sys
import subprocess
import os
from pathlib import Path

def run_make_command(target: str) -> None:
    """Run a make command with the given target.
    
    Args:
        target (str): The target to pass to the make command.
    """
    # Run the make command here with the target passed or default
    print(f"Running make with target: {target}")
    subprocess.check_call(['make', target])

def run_make_commands(targets: list[str]) -> None:
    """Run make commands with the given targets.
    
    Args:
        targets (list[str]): The targets to pass to the make command.
    """
    for target in targets:
        run_make_command(target)

if __name__ == '__main__':
    args = sys.argv[1:]
    cwd_orig = os.getcwd()
    makefile_dir = Path(os.path.dirname(__file__)).parent.parent
    os.chdir(makefile_dir)
    run_make_commands(args)
    os.chdir(cwd_orig)