"""Leakage-safe train/val/test splits.

Each Valencia simulation is projected through several torso geometries. Those copies share
the same atrial driver, so they must never be split across train and test.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def group_split(
    groups: np.ndarray, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with no group shared between parts."""
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval, test = next(outer.split(idx, groups=groups))
    if val_size <= 0:
        return trval, np.array([], dtype=int), test
    rel_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr, va = next(inner.split(trval, groups=groups[trval]))
    return trval[tr], trval[va], test
