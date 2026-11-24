# ML-Enabled Non-Cooperative Orbit Estimation and Anomaly Detection with LEO Satellite Signals

BSc Honours project in Computer Science, University of the Western Cape (2026).
Raeez Ahmed. Supervisor: Prof. Mehrdad Ghaziasghar. Co-supervisor: Prof. Christopher P. Thron.

This project asks how much of a low Earth orbit (LEO) satellite's pass can be estimated from the timing of its signal alone, with no help from the satellite's operator. A ground receiver measures how the arrival time of repeats in the signal changes during one pass, which gives the satellite's range rate (its Doppler). Machine learning models then use that range rate curve to:

1. **Estimate the pass:** eight labels describing where the satellite is, and how it is moving, at its closest approach to the station.
2. **Detect anomalies:** decide whether a pass was hit by a burst of extra noise.

All data is simulated with [orbdetpy](https://github.com/ut-astria/orbdetpy), which runs the [Orekit](https://www.orekit.org) astrodynamics library.

---

## Pipeline

```
Clean_Testing        simulate 10,000 satellite passes over one ground station
      |
Closest_Approach     find each pass's closest approach and its 8 labels
      |
Trajectory_GPU_v2    train and test LightGBM, Random Forest and a 1D CNN,
      |              on clean data and with added measurement noise
Solar_Flare          add noise bursts to half of the passes and train classifiers to find them
      |
Dashboard            Streamlit dashboard of all the results
```

### Simulation (`Final_V/Clean_Testing`)
- One ground station on the equator (latitude 0, longitude -2.7272 rad, altitude 0 m).
- 10,000 satellites. Each orbit is built so that the satellite passes the station with an elevation of 40 to 90 degrees at closest approach, with a semi-major axis 400 to 1,500 km above the Earth's equatorial radius and an eccentricity of at most 0.02 (`generate_params.py`).
- Each pass is propagated for 24 minutes with orbdetpy's full force model and sampled every 30 s while the satellite is above the horizon (`simv3.py`).
- The measurement is the delay difference between signal repeats 1 ms apart, which is proportional to the range rate.

### Labels (`Final_V/Closest_Approach`)
`closest_approach.py` repeats the simulation's propagation and finds the exact moment of closest approach. The eight labels are:

| Label | Type | Meaning |
|---|---|---|
| Altitude | km | height of the satellite above the Earth's surface |
| Elevation | deg | angle above the station's horizon |
| Folded azimuth | deg | angle from due north or due south, 0 to 90 |
| Angular rate | deg/s | how fast the satellite crosses the station's sky |
| Radial velocity | m/s | how fast the satellite's distance from the Earth's centre changes |
| Eastbound | yes/no | the satellite is moving east |
| Northbound | yes/no | the satellite is moving north |
| Passes east | yes/no | the satellite passes east of the station |

`reconstruct_check_v2.py` shows that the true values of the eight labels pin down the satellite's position and velocity at closest approach for 9,796 of the 10,000 satellites.

### Orbit estimation (`Final_V/Trajectory_GPU_v2`)
Each pass is aligned at the moment its range rate crosses zero and resampled every 30 s (`build_dataset.py`). LightGBM, a Random Forest and a 1D CNN are trained on 8,000 satellites and tested on the other 2,000 (split seed 42). The same models are then retrained with Gaussian noise of 0.5, 1, 2 and 4 m/s added to the range rate (`sweep/`).

### Noise bursts (`Final_V/Solar_Flare`)
Half of the passes get one burst of 5 m/s noise covering 30 to 70% of the pass. LightGBM and a Random Forest are trained to say whether a pass had a burst, using the readings and their first, second and third differences.

---

## Main results

Test set of 2,000 satellites, clean data. Guessing means always predicting the average (numbers) or the more common answer (yes/no).

| Label | Best model | Guessing |
|---|---|---|
| Altitude (RMSE) | 13.6 km (CNN) | 318.3 km |
| Eastbound (accuracy) | 95.0% (LightGBM, Random Forest) | 85.3% |
| Northbound (accuracy) | 50.7% (CNN) | 50.7% |
| Passes east (accuracy) | 62.7% (LightGBM) | 51.7% |

Northbound cannot be learned from a station on the equator, because a pass and its reflected copy across the equator give the same range rate curve.

Noise burst detection: LightGBM is correct on 95.0% of test passes (guessing: 50.4%) and finds 91.8% of the bursts.

Full results are in `Final_V/Trajectory_GPU_v2/results_summary.csv` and `Final_V/Solar_Flare/output/flare_results.csv`, and in the dashboard.

---

## Repository layout

```
blind_orbit_ml/
├── README.md
├── requirements.txt
├── Archive/                      earlier experiments and development notes
└── Final_V/                      the final version of the project
    ├── Clean_Testing/            simulation: generate_params.py, simv3.py, output/
    ├── Closest_Approach/         closest_approach.py, reconstruct_check_v2.py, output/
    ├── Trajectory_GPU_v2/        orbit estimation models, noise sweep, results_summary.csv
    ├── Solar_Flare/              noise burst detection, burst_check.py, output/
    ├── Dashboard/                app.py (Streamlit)
    ├── Noise_Testing/            earlier noise runs (simv4.py), kept for reference
    └── master file v2.ipynb
```

The result files are included, so the dashboard runs without re-running the pipeline. The trained models from an earlier six-label run in `Clean_Testing/output/models/` are not included, because some are over GitHub's 100 MB file limit.

`Noise_Testing` is an earlier stage. Its imports still use the old `V3.Noise_Testing` paths, and the final noise sweep is done by `Trajectory_GPU_v2/run_all.py` instead.

---

## Setup

Tested on Windows 11 with Python 3.13.8.

1. Install **Java 17** (for example Eclipse Temurin) and set `JAVA_HOME` to it. orbdetpy needs it to run Orekit. Only the simulation and `closest_approach.py` use orbdetpy.
2. Create a virtual environment and install the packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

orbdetpy also needs an Orekit data folder. If its automatic download fails, replace the `orekit-data` folder with a complete bundle.

The CNN uses a GPU if PyTorch finds one, and the CPU otherwise. `Final_V/Trajectory_GPU_v2/COLAB.md` explains how to run that stage on Google Colab.

---

## Running

Run the stages in order. Each script is run from its own folder.

```powershell
# 1. Simulation (also trains the earlier six-label models)
cd Final_V\Clean_Testing
python run_clean.py

# 2. Closest approach labels
cd ..\Closest_Approach
python closest_approach.py
python reconstruct_check_v2.py output\closest_approach.csv

# 3. Orbit estimation, clean and with noise
cd ..\Trajectory_GPU_v2
python run_all.py --sigmas 0 0.5 1 2 4

# 4. Noise burst detection
cd ..\Solar_Flare
python solar_flare.py
python burst_check.py

# 5. Dashboard
cd ..\Dashboard
streamlit run app.py
```

`run_all.py` skips any step whose output already exists, so it can be stopped and restarted. `burst_check.py` rebuilds the burst data, checks it against the saved labels and plots one example pass.

---

## References

- orbdetpy: https://github.com/ut-astria/orbdetpy
- Orekit: https://www.orekit.org
