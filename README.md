# ML-Enabled Blind Timing-Based Orbit Estimation and Anomaly Detection with LEO Satellite Signals

*Data Science / Applied ML Capstone Project — Satellite Instrumentation Focus*

This repository builds a **simulation pipeline** that generates realistic timing
(delay) measurements of Low Earth Orbit (LEO) satellite passes, as the data foundation
for a machine-learning system that estimates a satellite's orbit — and detects anomalies —
from timing alone, without knowing anything about the radio signal itself.

---

## 1. Motivation

Modern RF and satellite test systems (monitoring receivers, spectrum analyzers, channel
sounders, "signals-of-opportunity" testbeds) increasingly face a practical problem:
**valuable satellite downlink signals are often not publicly documented.** An instrument
may not know the modulation format or waveform structure in advance.

However, even when the waveform is unknown, it is usually still possible to extract
**timing observables** — for example, the **signal delay** (how long the signal takes to
travel from the satellite to the receiver). Because the satellite is moving, this delay
changes throughout an overpass, and that changing delay **encodes the geometry of the
satellite's trajectory**.

This project reframes that idea as an **applied machine-learning task**: design a system
that estimates satellite orbital parameters directly from delay measurements, and detects
anomalous events in the delay time series — using **simulated, publicly reproducible data**.

---

## 2. The Core Idea (Problem Statement)

We work with simulated delay measurements `τ(t_k)` taken during a *portion* of a LEO
overpass. Delay is tied to geometry by a simple relationship:

```
τ(t) = ρ(t) / c          # delay = range / speed of light

ρ(t) = || r_sat(t) − r_rx(t) ||   # range = distance between satellite and receiver
```

- `ρ(t)` — the **range**: straight-line distance from satellite to ground receiver at time `t`
- `c`    — the speed of light (299,792,458 m/s)
- `r_sat`, `r_rx` — the 3D positions of the satellite and the ground receiver

The waveform itself is treated as **unknown** — only the timing matters. Because of orbital
motion, the delay time series has **strong curvature**, is **sensitive to orbital parameters**
(altitude, inclination, pass geometry), contains **measurement noise**, and may contain
**anomalies** (maneuvers, noise bursts, missing data).

### Goals
- **Primary:** estimate satellite orbital parameters from the delay time series of a partial overpass.
- **Secondary:** detect and classify anomalous events — a small orbital maneuver (ΔV), a temporary noise burst, or a data dropout.

---

## 3. Approach: Simulation-First, Publicly Reproducible

All data is generated with the [**orbdetpy**](https://github.com/ut-astria/orbdetpy)
orbital propagation toolkit (a Python wrapper around the Orekit astrodynamics library).
orbdetpy acts purely as the **physics/measurement engine** — it is *not* the ML framework.

The simulation models:
- a realistic **LEO satellite trajectory**,
- a **fixed ground receiver**,
- the **geometric range and delay** between them,
- realistic **partial-overpass visibility** (the satellite is only observed while above the horizon),
- controlled **Gaussian measurement noise**.

---

## 4. Current Status

| Phase | Status | What it delivers |
|------|--------|------------------|
| **V1.0** | ✅ Done | One nominal pass, one station, RANGE only — raw measurement objects |
| **V1.1** | ✅ Done | Extract time, range, and delay (`delay = range / c`) into clean lists |
| **V1.2** | ✅ Done | Plot range & delay vs. time; save data to CSV |
| **V1.3** | ✅ Done | Save run metadata to JSON; write all outputs into an `output/` folder |
| **V1.4** | ✅ Done | **Realistic LEO orbit + physically realistic pass sampling** (see §6) |
| **V2.0** | ⏭ Next | Generate **many** nominal passes by varying orbital elements (dataset building) |
| **V3.0** | Planned | Inject anomalies: ΔV maneuver, noise burst, data dropout |
| **V4.0** | Planned | Baseline + ML models for orbit estimation and anomaly detection |

The single script `simulationv1.py` currently produces **one fully realistic nominal pass**.

---

## 5. Repository Structure

```
blind_orbit_ml/
├── simulationv1.py        # MAIN pipeline (current version: V1.4)
├── output/                # Generated per run (git-ignored data products)
│   ├── nominal_pass_001.csv    # time, range, delay, time-since-start
│   ├── nominal_pass_001.json   # run metadata (orbit + pass summary)
│   └── nominal_pass_001.png    # range & delay vs. time plots
├── test.py                # orbdetpy's reference example (kept for reference)
├── step1_inspect.py       # helper: list orbdetpy's top-level API
├── step1_help.py          # helper: print signatures/docs of key functions
├── step1-measurements.py  # helper: list available MeasurementType values
├── .gitignore             # ignores .venv, output artifacts, Python caches
└── README.md              # this file
```

The `step1_*.py` files are small inspection scripts used while learning the orbdetpy API.
`test.py` is the upstream orbdetpy example, kept as a reference for features (orbit
determination, maneuvers, multiple stations) that later versions will draw from.

---

## 6. What `simulationv1.py` Does (V1.4 in detail)

### a. Build a realistic LEO orbit from orbital elements
Instead of hard-coding a raw position/velocity vector, the orbit is described with
**human-readable Keplerian orbital elements** based on a real ISS-like satellite, then
converted to the precise state vector orbdetpy needs via `elem_to_pv()`:

| Element | Value | Meaning |
|--------|-------|---------|
| Semi-major axis | 6,798,000 m | ≈ 416 km altitude (true LEO) |
| Eccentricity | 0.0006 | near-circular |
| Inclination | 51.6° | real ISS orbital tilt |
| RAAN | 60° | orientation of the orbital plane |
| Arg. of perigee | 0° | orientation within the plane |
| Mean anomaly | 0° | starting point along the orbit |

Frame: **EME2000** (J2000 inertial). Verified output: altitude 415.9 km, speed 7.662 km/s —
both correct for LEO.

### b. Observe from a fixed ground station
A single ground station, **"Maui"** (Haleakalā: 20.7° N, 156.3° W, 3059 m altitude),
measures **RANGE** with 10 m of Gaussian measurement noise.

### c. Capture the pass at a realistic, fixed cadence
orbdetpy only returns a measurement when the satellite is **above the station's horizon**.
The script propagates over a 20-minute search window at a **fixed 30-second sampling
cadence** (modeling a real station's steady sampling rate) and keeps every visible point.

The **number of records is not forced** — it falls out naturally from the pass geometry
(a higher, longer pass yields more points). For this orbit/station the result is:

- Pass start (AOS): `00:01:36`  →  Pass end (LOS): `00:08:06`
- Duration: **~6.5 minutes visible** (consistent with typical LEO overpass length)
- **14 records**, with range tracing a smooth U-shape: 1,848 km → 1,237 km (closest
  approach) → 1,856 km

### d. Convert, report, and save
For each measurement the script computes `delay = range / c`, prints a formatted table,
and writes three artifacts into `output/` (CSV data, JSON metadata, PNG plot).

---

## 7. Output Files

### `nominal_pass_001.csv`
| Column | Description |
|--------|-------------|
| `UTC Time` | Measurement time (UTC) |
| `Range (m)` | Satellite–station distance in metres |
| `Delay (s)` | `Range / c` — the timing observable the ML model will use |
| `time_since_start_s` | Seconds since the first measurement of the pass |

### `nominal_pass_001.json` (metadata)
Records exactly how the sample was generated, so every pass is self-documenting and
reproducible: `sample_id`, satellite/station names, measurement type, detected
`prop_start_utc`/`prop_end_utc`, `sample_cadence_s`, `pass_duration_s`, `num_records`,
`range_error_m`, the full `prop_initial_state` vector, and the `orbital_elements` block.
This metadata is what **V2.0 will rely on** when varying orbits to build a labeled dataset.

---

## 8. Environment & Setup

This project depends on a Java runtime (orbdetpy runs Orekit under the hood via a Java
backend) and a Python virtual environment.

**Verified working environment**
- **OS:** Windows 11
- **Python:** 3.13.8 (in a local `.venv`, not committed to git)
- **Java:** OpenJDK 17.0.8 (Eclipse Temurin), with `JAVA_HOME` set to the JDK 17 folder
- **orbdetpy:** 2.1.0 — plus `matplotlib` for plotting (`csv`, `json`, `os`, `math` are standard library)

**Orekit data note:** orbdetpy requires an Orekit data bundle (Earth-orientation
parameters, leap seconds, etc.). The automatic download partially failed on first setup
(connection reset on `finals.all`, `finals2000A.all`, `tai-utc.dat`); these were resolved by
manually replacing the `orekit-data` folder with a complete bundle.

### Install (fresh machine)
```powershell
# from the project root
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
pip install orbdetpy matplotlib
```
Ensure JDK 17 is installed and `JAVA_HOME` points to it.

---

## 9. How to Run
```powershell
.venv\Scripts\Activate.ps1
python simulationv1.py
```
This prints the range/delay table, writes the three files into `output/`, and opens a
matplotlib window with the range and delay plots. (For a headless run that skips the
window, set the environment variable `MPLBACKEND=Agg`.)

---

## 10. Roadmap to the ML System

The simulation pipeline feeds the ML tasks defined in the project outline:

- **V2.0 — Dataset generation.** Vary the orbital elements (altitude, inclination, pass
  geometry) to produce many labeled nominal passes. Supports **Task A: Orbit Estimation**
  (`θ̂ = f_φ(τ(t₁),…,τ(t_K))`) with candidate models such as gradient-boosted regression on
  engineered delay features, 1D CNNs, or sequence models.
- **V3.0 — Anomalies.** Add ΔV maneuvers, noise bursts, and data dropouts, supporting
  **Task C: Anomaly Detection** (classify passes from the residuals of a fitted model).
- **V4.0 — Models & baselines.** Train ML models and compare against the required baselines:
  direct nonlinear least squares, curvature thresholding, and variance/gap rules.
  **Task B: Orbit Refinement** (nonlinear least squares) refines ML predictions.

**Planned additional dependencies** for these phases: NumPy, SciPy, pandas, scikit-learn,
and optionally PyTorch or TensorFlow for sequence models. No specialized DSP libraries are
needed — delay sequences are treated as numerical time-series data.

---

## 11. Key Design Decisions

- **orbdetpy is the simulator, not the ML framework.** It generates ground-truth physics; ML lives in later phases.
- **Orbit defined by elements, not raw vectors.** Realistic, reproducible, and easy to vary for V2.0.
- **Horizon visibility is modeled, not bypassed.** orbdetpy naturally returns only above-horizon points — this *is* the partial-overpass arc the project calls for.
- **Fixed cadence, variable record count.** Mirrors a real ground station and preserves pass-duration as a genuine signal; variable sequence length is handled later at the ML-preprocessing stage.

---

## 12. References
- orbdetpy — https://github.com/ut-astria/orbdetpy
- Orekit (underlying astrodynamics library) — https://www.orekit.org
