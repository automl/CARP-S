from __future__ import annotations

import warnings
from pathlib import Path

import pkg_resources
from omegaconf import DictConfig


def _check(p: str | Path) -> None:
    """Check requirements specified in p

    Parameters
    ----------
    p : str | Path
        Path to requirements. Txt file.

    Raises
    ------
    RuntimeError
        When the requirement is not installed.
    Warning
        When there is a version conflict.
    """
    p = Path(p)
    if p.is_file():
        with open(p, "r") as file:
            requirements = file.readlines()
        requirements = [r.strip() for r in requirements]

        try:
            pkg_resources.require(requirements)
        except pkg_resources.DistributionNotFound as error:
            error_msg = str(error)
            msg = f"{error_msg}. Please install all necessary requirements with\n"\
                    f"\t>>>>>>>>>> pip install -r {p}"
            raise RuntimeError(msg)
        except pkg_resources.VersionConflict as error:
            error_msg = str(error)
            msg = f"Version Conflict: {error_msg} for {p}. Resolve manually. If it is about YAHPO and ConfigSpace, run the following "\
                    "to make YAHPO compatible with newest ConfigSpace:\n"\
                    f"\t>>>>>>>>>> python {p.parent.parent.parent.parent / 'scripts/patch_yahpo_configspace.py'}"
            warnings.warn(msg)
       
        

def check_requirements(cfg: DictConfig) -> None:
    """Check whether requirements are satisified for benchmark and optimizer

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration
    """
    p_base = Path(__file__).parent.parent.parent / "container_recipes"
    req_file_benchmark = p_base / "benchmarks" / cfg.benchmark_id / f"{cfg.benchmark_id}_requirements.txt"
    req_file_optimizer = p_base / "optimizers" / cfg.optimizer_container_id / f"{cfg.optimizer_container_id}_requirements.txt"

    _check(req_file_benchmark)
    _check(req_file_optimizer)
