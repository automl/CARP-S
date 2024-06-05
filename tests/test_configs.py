from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("carps/configs/optimizer", "optimizer_id"),
        ("carps/configs/problem", "problem_id"),
    ],
)
def test_unique_ids(path, key):
    path = Path(path)
    paths = list(path.glob("**/*.yaml"))
    values = []
    for p in paths:
        cfg = OmegaConf.load(p)
        value = cfg.get(key)
        values.append(value)

    assert len(set(values)) == len(values), (
        "Duplicate " f"{key}, they need to have a unique name: {[k for k, v in Counter(values).items() if v > 1]}"
    )
