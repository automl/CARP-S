"""Get the container directory for HPOBench."""

from __future__ import annotations

from hpobench.config import HPOBenchConfig


def get_container_dir() -> str:
    """Get the container directory for HPOBench.

    Returns:
        str: The container directory for HPOBench.
    """
    config = HPOBenchConfig()
    return config.container_dir


if __name__ == "__main__":
    print(get_container_dir())
