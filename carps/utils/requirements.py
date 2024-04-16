from __future__ import annotations
from omegaconf import DictConfig
from rich import print as printr
from pathlib import Path

def check_requirements(cfg: DictConfig) -> None:
    printr(cfg.benchmark_id)
    printr(cfg.optimizer_container_id)

    p_base = Path(__file__).parent.parent / "container_recipes"
    req_file_benchmark = p_base / "benchmarks" / cfg.benchmark_id / f"{cfg.benchmark_id}_requirements.txt"
    req_file_optimizer = p_base / "optimizers" / cfg.optimizer_container_id / f"{cfg.optimizer_container_id}_requirements.txt"
