# Data

Nothing here is committed: EDGAR data is for registered users only.

## Source

**EDGAR – Simulation of Atrial Rotors (Valencia_sim_08-01-2014)**, from the
University of Utah's Experimental Data and Geometric Analysis Repository:
https://edgar.sci.utah.edu. It was provided by ITACA (Universitat Politècnica de València).

Every page on EDGAR requires a login. Register at https://edgar.sci.utah.edu/register/.

Per the 2025 Frontiers study (doi:10.3389/fphys.2025.1720244), the data uses:
- a bilayer atrial model with 2,048 nodes
- 10 MRI-based torso geometries per electrogram
- a simulated 64-electrode vest

Cite Aras et al., *J Electrocardiol* 48(6), 2015 (doi:10.1016/j.jelectrocard.2015.08.008)
and Pedrón-Torrecilla et al., 2016.

## Layout

```
data/
  raw/         # files exactly as downloaded from EDGAR (unzipped)
  processed/   # valencia.npz written by the loader (BSPMRecordings.save)
```

## TLS note

edgar.sci.utah.edu does not send its intermediate certificate, so `curl`, `requests`
and `wget` fail verification by default. Never disable verification. Build a bundle
that includes the missing intermediates (they are committed in `certs/`) instead:

```bash
scripts/edgar_ca_bundle.sh   # writes .edgar-ca-bundle.pem
curl --cacert .edgar-ca-bundle.pem https://edgar.sci.utah.edu/
```
