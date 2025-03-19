from __future__ import annotations
from hpobench.config import HPOBenchConfig

def get_container_dir() -> str:
    config = HPOBenchConfig()
    return config.container_dir

if __name__ == "__main__":
    print(get_container_dir())
