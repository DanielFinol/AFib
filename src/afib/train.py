"""Train the driver-region CNN.

    afib-train --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from afib.data.dataset import BSPMRecordings, WindowDataset, make_windows
from afib.data.splits import group_split
from afib.data.synthetic import make_synthetic
from afib.evaluate import summarize
from afib.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_recordings(cfg: dict, seed: int) -> BSPMRecordings:
    d = cfg["data"]
    regions = cfg["labels"]["regions"]
    if d["source"] == "synthetic":
        return make_synthetic(regions, fs_hz=d["fs_hz"], seed=seed)
    if d["source"] == "edgar":
        rec = BSPMRecordings.load(d["processed_path"])
        if rec.regions != regions:
            raise ValueError(f"config regions {regions} != data regions {rec.regions}")
        return rec
    raise ValueError(f"unknown data source: {d['source']}")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rec = load_recordings(cfg, seed)
    s = cfg["split"]
    groups = getattr(rec, s.get("group_by", "simulation_id"))
    tr, va, te = group_split(groups, s["test_size"], s["val_size"], seed)
    d = cfg["data"]
    win = lambda r: make_windows(r, d["window_s"], d["hop_s"], d["normalize"])
    Xtr, ytr, _ = win(rec.subset(tr))
    Xva, yva, rva = win(rec.subset(va))
    Xte, yte, rte = win(rec.subset(te))

    t = cfg["train"]
    model = build_model(cfg["model"], n_leads=Xtr.shape[1], n_classes=len(rec.regions)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=t["batch_size"], shuffle=True)

    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, history = -1.0, []
    for epoch in range(t["epochs"]):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        val = summarize(yva, predict(model, Xva, t["batch_size"], device), rva, rec.regions)
        history.append({"epoch": epoch, "loss": total / len(ytr), **val})
        print(
            f"epoch {epoch:3d}  loss {total / len(ytr):.4f}  "
            f"val rec bal-acc {val['recording_balanced_accuracy']:.3f}"
        )
        if val["recording_balanced_accuracy"] > best_val:
            best_val = val["recording_balanced_accuracy"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test = summarize(yte, predict(model, Xte, t["batch_size"], device), rte, rec.regions)
    (out_dir / "metrics.json").write_text(json.dumps({"test": test, "history": history}, indent=2))
    print(json.dumps({k: v for k, v in test.items() if "confusion" not in k}, indent=2))
    return test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
