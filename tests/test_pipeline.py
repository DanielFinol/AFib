import numpy as np
import pytest

from afib.data import group_split, make_windows
from afib.data.dataset import BSPMRecordings
from afib.data.synthetic import make_synthetic

REGIONS = ["PV", "RAA", "RA_free_wall"]


def test_group_split_never_shares_simulations():
    rec = make_synthetic(REGIONS, n_simulations_per_region=5, n_torsos=10, duration_s=1)
    tr, va, te = group_split(rec.simulation_id, 0.2, 0.1, seed=0)
    s = [set(rec.simulation_id[i]) for i in (tr, va, te)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    assert len(tr) + len(va) + len(te) == len(rec)


def test_windows_shape_and_labels():
    rec = make_synthetic(REGIONS, n_simulations_per_region=1, n_torsos=2, duration_s=3, fs_hz=100)
    X, y, r = make_windows(rec, window_s=1.0, hop_s=1.0)
    assert X.shape == (len(rec) * 3, 64, 100)
    assert np.all(y == rec.labels[r])


def test_save_load_roundtrip(tmp_path):
    rec = make_synthetic(REGIONS, n_simulations_per_region=1, n_torsos=2, duration_s=1)
    rec.save(tmp_path / "x.npz")
    back = BSPMRecordings.load(tmp_path / "x.npz")
    assert back.regions == REGIONS and len(back) == len(rec)
    np.testing.assert_allclose(back.signals[0], rec.signals[0])


def test_cnn_learns_synthetic(tmp_path):
    pytest.importorskip("torch")
    from afib.train import run

    cfg = {
        "seed": 0,
        "data": {"source": "synthetic", "fs_hz": 100, "window_s": 1.0, "hop_s": 1.0,
                 "normalize": "per_window"},
        "split": {"group_by": "simulation_id", "test_size": 0.25, "val_size": 0.15},
        "labels": {"regions": REGIONS},
        "model": {"name": "bspm_cnn1d", "channels": [16, 32], "kernel_size": 5, "dropout": 0.1},
        "train": {"epochs": 8, "batch_size": 32, "lr": 3e-3, "weight_decay": 0.0,
                  "out_dir": str(tmp_path)},
    }
    test = run(cfg)
    assert test["recording_balanced_accuracy"] > 0.8
