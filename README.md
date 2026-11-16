# AFib: locating AF drivers from body-surface potentials

Goal: train a small CNN on the **EDGAR Valencia atrial rotor simulations**
(`Valencia_sim_08-01-2014`) to predict **which atrial region drives the AF**, using only
the 64-lead body-surface potential map (BSPM).

## Baseline to compare against

Cámara-Vázquez et al., *Non-invasive Estimation of Atrial Fibrillation Driver Position With
Convolutional Neural Networks and Body Surface Potentials*, Front. Physiol. 12:733449 (2021).
[PMC8552066](https://pmc.ncbi.nlm.nih.gov/articles/PMC8552066/)

- Classes: posterior left atrial wall, left and right pulmonary veins, right atrial
  appendage, right atrial free wall.
- Accuracy is 0.75 at 5 dB SNR and 0.93 at 20 dB or more, when windows are treated as
  independent.
- It falls to **≤0.52 when whole AF models are held out**.

That drop is the number to beat honestly. Every split here is grouped by simulation, so
the 10 torso projections of one electrogram never appear in both train and test. See
`src/afib/data/splits.py`.

## Layout

```
configs/baseline.yaml      experiment config (data, split, labels, model, training)
src/afib/data/
  loaders.py               read raw EDGAR .mat files; `afib-inspect` lists their contents
  dataset.py               BSPMRecordings container, windowing, normalisation
  splits.py                group-wise train/val/test split (no simulation leakage)
  synthetic.py             toy data so the pipeline runs before the real data arrives
src/afib/models/cnn.py     1D CNN: 64 leads as channels, convolved over time
src/afib/train.py          training loop and checkpointing (`afib-train`)
src/afib/evaluate.py       per-window and per-recording (vote) metrics
tests/                     pipeline tests on synthetic data
certs/, scripts/           TLS workaround for edgar.sci.utah.edu (see data/README.md)
```

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run afib-train --config configs/baseline.yaml   # data.source: synthetic by default
```

## Roadmap

1. **Get the data.** Register on EDGAR and put the Valencia files in `data/raw/`
   (see `data/README.md`).
2. **Inspect them.** Run `afib-inspect data/raw/`, then implement `load_valencia()` to
   write `data/processed/valencia.npz`. If drivers are given as nodes on the 2,048-node
   mesh, add a node-to-region map.
3. **Baseline.** Set `data.source: edgar`, then report per-recording balanced accuracy
   with grouped splits. Add noise at 5/10/20 dB SNR to match the paper.
4. **Improve.** Try a 2D CNN on the vest laid out as a grid, spectral (dominant-frequency)
   inputs, and leave-one-torso-out tests for robustness to geometry.
