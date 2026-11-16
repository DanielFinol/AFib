"""Reading the raw EDGAR Valencia files.

The exact file layout is not known until the dataset is downloaded (EDGAR requires an
account). Step 1 is to inspect the files with:

    afib-inspect data/raw/

which lists every variable in every .mat file with its shape. Then fill in
``load_valencia`` to map those variables onto ``BSPMRecordings``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from afib.data.dataset import BSPMRecordings


def read_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Read a .mat file of any version (v7.3 files are HDF5)."""
    import h5py
    import scipy.io

    path = Path(path)
    try:
        d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except NotImplementedError:
        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            f.visititems(
                lambda name, obj: out.__setitem__(name, obj[()])
                if isinstance(obj, h5py.Dataset)
                else None
            )
        return out


def describe(root: str | Path) -> None:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        print(f"no files under {root}")
        return
    for p in files:
        print(f"\n== {p.relative_to(root)} ({p.stat().st_size / 1e6:.1f} MB)")
        if p.suffix.lower() != ".mat":
            continue
        try:
            for k, v in read_mat(p).items():
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", type(v).__name__)
                print(f"   {k:30s} shape={shape} dtype={dtype}")
        except Exception as e:  # noqa: BLE001 - inspection tool, report and move on
            print(f"   could not read: {e}")


def load_valencia(raw_dir: str | Path, regions: list[str], fs_hz: float) -> BSPMRecordings:
    """Map the raw Valencia files to BSPMRecordings.

    TODO once the data is inspected:
      * which variable holds the 64-lead vest potentials (leads x samples)?
      * where is the driver location stored (node index on the 2,048-node atrial mesh,
        region name, or only in the file name)?
      * how are the 10 torso geometries per electrogram indexed?
    If drivers are given as mesh nodes, add a node -> region lookup in ``regions.py``.
    """
    raise NotImplementedError(
        "Inspect the raw files with `afib-inspect data/raw/` and implement load_valencia()."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="List variables in raw EDGAR files")
    ap.add_argument("root", nargs="?", default="data/raw")
    describe(ap.parse_args().root)


if __name__ == "__main__":
    main()
