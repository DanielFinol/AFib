"""Toy BSPM generator so the pipeline can run before the EDGAR data is available.

Each region gets a characteristic spatial pattern across the 64 leads and a dominant
frequency; torso geometry is mimicked by a random per-lead gain. This is NOT physiology,
only a smoke test that the model can learn a spatial-spectral signature.
"""

from __future__ import annotations

import numpy as np

from afib.data.dataset import BSPMRecordings


def make_synthetic(
    regions: list[str],
    n_simulations_per_region: int = 8,
    n_torsos: int = 10,
    n_leads: int = 64,
    duration_s: float = 4.0,
    fs_hz: float = 500.0,
    seed: int = 0,
) -> BSPMRecordings:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs_hz)) / fs_hz
    lead_pos = np.linspace(0, 2 * np.pi, n_leads, endpoint=False)
    signals, labels, sims, torsos = [], [], [], []
    sim = 0
    for r in range(len(regions)):
        centre = 2 * np.pi * r / len(regions)
        for _ in range(n_simulations_per_region):
            f = 5.0 + r * 0.6 + rng.normal(0, 0.2)        # dominant frequency, Hz
            phase = rng.uniform(0, 2 * np.pi)
            spatial = np.exp(np.cos(lead_pos - centre - rng.normal(0, 0.15)) * 2.0)
            source = np.sin(2 * np.pi * f * t + phase)
            for k in range(n_torsos):
                gain = rng.lognormal(0, 0.2, size=n_leads)
                x = (spatial * gain)[:, None] * source[None, :]
                x += rng.normal(0, 0.5, size=x.shape)
                signals.append(x.astype(np.float32))
                labels.append(r)
                sims.append(sim)
                torsos.append(k)
            sim += 1
    return BSPMRecordings(
        signals=signals,
        labels=np.array(labels),
        simulation_id=np.array(sims),
        torso_id=np.array(torsos),
        regions=list(regions),
        fs_hz=fs_hz,
    )
