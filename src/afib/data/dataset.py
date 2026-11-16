"""In-memory representation of BSPM recordings and windowing into CNN samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BSPMRecordings:
    """A set of body-surface recordings.

    Attributes:
        signals: list of arrays, each (n_leads, n_samples). Lengths may differ.
        labels: int array (n_recordings,), index into ``regions``.
        simulation_id: int array (n_recordings,), which atrial simulation produced it.
        torso_id: int array (n_recordings,), which torso geometry it was projected through.
        regions: names of the label classes.
        fs_hz: sampling rate.
    """

    signals: list[np.ndarray]
    labels: np.ndarray
    simulation_id: np.ndarray
    torso_id: np.ndarray
    regions: list[str]
    fs_hz: float

    def __post_init__(self) -> None:
        n = len(self.signals)
        for name in ("labels", "simulation_id", "torso_id"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, expected {n}")
        if n and len({s.shape[0] for s in self.signals}) != 1:
            raise ValueError("all recordings must have the same number of leads")

    def __len__(self) -> int:
        return len(self.signals)

    def subset(self, idx: np.ndarray) -> BSPMRecordings:
        idx = np.asarray(idx)
        return BSPMRecordings(
            signals=[self.signals[i] for i in idx],
            labels=self.labels[idx],
            simulation_id=self.simulation_id[idx],
            torso_id=self.torso_id[idx],
            regions=self.regions,
            fs_hz=self.fs_hz,
        )

    def save(self, path: str | Path) -> None:
        """Save to .npz. Signals are stored as an object array to allow ragged lengths."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sig = np.empty(len(self.signals), dtype=object)
        sig[:] = self.signals
        np.savez_compressed(
            path,
            signals=sig,
            labels=self.labels,
            simulation_id=self.simulation_id,
            torso_id=self.torso_id,
            regions=np.array(self.regions),
            fs_hz=self.fs_hz,
        )

    @classmethod
    def load(cls, path: str | Path) -> BSPMRecordings:
        z = np.load(path, allow_pickle=True)
        return cls(
            signals=list(z["signals"]),
            labels=z["labels"].astype(int),
            simulation_id=z["simulation_id"].astype(int),
            torso_id=z["torso_id"].astype(int),
            regions=[str(r) for r in z["regions"]],
            fs_hz=float(z["fs_hz"]),
        )


def _normalize(w: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return w
    if mode == "per_window":
        return (w - w.mean()) / (w.std() + 1e-8)
    if mode == "per_lead":
        return (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
    raise ValueError(f"unknown normalize mode: {mode}")


def make_windows(
    rec: BSPMRecordings, window_s: float, hop_s: float, normalize: str = "per_window"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut every recording into fixed-length windows.

    Returns:
        X: (n_windows, n_leads, window_len) float32
        y: (n_windows,) int64 labels
        rec_idx: (n_windows,) index of the source recording, for per-recording evaluation
    """
    win = round(window_s * rec.fs_hz)
    hop = round(hop_s * rec.fs_hz)
    if win <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    xs, ys, ri = [], [], []
    for i, s in enumerate(rec.signals):
        for start in range(0, s.shape[1] - win + 1, hop):
            xs.append(_normalize(s[:, start : start + win], normalize))
            ys.append(rec.labels[i])
            ri.append(i)
    if not xs:
        raise ValueError(f"no recording is at least {window_s}s long")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ri, dtype=np.int64),
    )


class WindowDataset:
    """Minimal torch-compatible dataset over pre-cut windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]
