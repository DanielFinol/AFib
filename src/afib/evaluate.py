"""Metrics, reported both per window and per recording (majority vote over windows)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def per_recording_vote(probs: np.ndarray, rec_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average window probabilities per recording. Returns (recording ids, predicted class)."""
    recs = np.unique(rec_idx)
    preds = np.array([probs[rec_idx == r].mean(axis=0).argmax() for r in recs])
    return recs, preds


def summarize(
    y_true: np.ndarray, probs: np.ndarray, rec_idx: np.ndarray, regions: list[str]
) -> dict:
    y_pred = probs.argmax(axis=1)
    recs, rec_pred = per_recording_vote(probs, rec_idx)
    rec_true = np.array([y_true[rec_idx == r][0] for r in recs])
    labels = list(range(len(regions)))
    return {
        "window_accuracy": float(accuracy_score(y_true, y_pred)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recording_accuracy": float(accuracy_score(rec_true, rec_pred)),
        "recording_balanced_accuracy": float(balanced_accuracy_score(rec_true, rec_pred)),
        "recording_confusion": confusion_matrix(rec_true, rec_pred, labels=labels).tolist(),
        "regions": regions,
    }
