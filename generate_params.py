# =============================================================================
# generate_params.py  –  Randomized orbital parameters + ECEF state vectors
# =============================================================================
#
# PURPOSE
# -------
# Generate randomized realistic LEO Keplerian orbits for batch input into
# simulate_doppler_pass() in simulationv2.py, and convert each orbit into its
# initial state vector in the EARTH-FIXED (ECEF / ITRF) frame via orbdetpy.
#
# Two CSVs are written, one row per satellite (aligned by sat_idx):
#
#   1. random_orbital_parameters.csv – six Keplerian elements + fixed receiver
#                                       settings + simulation settings.
#                                       (input for simulationv2.py)
#
#   2. initial_state_vectors.csv     – [x, y, z, vx, vy, vz] in the ECEF frame,
#                                       one satellite per row.
#                                       (labels for the ML stage)
#
# FRAME NOTE
# ----------
# Keplerian elements are only meaningful in an INERTIAL frame, so elem_to_pv is
# called in EME2000 (J2000 inertial / ECI).  The resulting state is then
# transformed to ECEF (ITRF) with transform_frame().  The ECEF velocity differs
# from the ECI velocity by the Earth-rotation term (omega x r) — this is handled
# correctly by transform_frame, which is why we convert the full 6-vector.
#
# YIELD NOTE
# ----------
# Only a fraction of random orbits actually pass over Maui within the search
# window (random phase + full inclination range, and inclinations below the
# station latitude ~20.7 deg can never pass overhead).  To end up with many
# satellites that DO produce a pass, two levers are used: a generous satellite
# count here, and a wide search window in simulationv2.py.
#
# SAMPLING RANGES (full realistic LEO)
# ------------------------------------
#   sma  [m]   : 6_571_000 – 7_371_000   (~200–1000 km altitude)
#   ecc        : 0.0 – 0.05              (near-circular LEO)
#   inc  [rad] : 0 – pi                  (full inclination range)
#   raan/argp/anom [rad] : 0 – 2pi       (uniform)
# =============================================================================

import math
import csv
import os
import random

from orbdetpy import Frame, PositionAngle
from orbdetpy.conversion import get_J2000_epoch_offset, elem_to_pv, transform_frame


# Fixed receiver location (Maui ground station)
STATION_NAME = "Maui"
STATION_LAT  = 0.3614   # [rad]
STATION_LON  = -2.7272  # [rad]
STATION_ALT  = 3059.0   # [m]

# Simulation settings (written into the params CSV, read back by simulationv2.py)
DELTA_T   = 0.001   # pulse-pair separation [s], used to scale range-rate into a delay
CADENCE_S = 30.0    # measurement cadence [s]

# 1-sigma RANGE-RATE measurement noise [m/s].
# Currently 0.0 -- noise is deliberately OFF while the physics is being
# validated. With clean data the Doppler curve should be smooth: negative on
# approach, zero at closest approach, positive on recession. Once that shape is
# confirmed, set this to a realistic receiver value (e.g. 0.1-1.0 m/s) to model
# measurement error.
RANGE_RATE_ERROR_MS = 0.0

# Reference epoch at which the orbital elements are defined
EPOCH_STR = "2019-05-01T00:00:00"

# ECEF frame: ITRF via CIO, 2010 conventions, accurate Earth-orientation params
ECEF_FRAME = Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP

# --- Sampling bounds: realistic, numerically stable LEO range ---
# Perigee = sma * (1 - ecc) must stay well above the atmosphere, or Orekit's
# integrator fails ("minimal step size reached"). The floor below keeps perigee
# altitude above ~330 km even at max eccentricity — and matches real operational
# LEO (ISS ~420 km, Starlink ~550 km); orbits much lower than this decay in days.
SMA_MIN, SMA_MAX = 6_778_000.0, 7_578_000.0   # ~400–1200 km altitude
ECC_MIN, ECC_MAX = 0.0, 0.01                  # near-circular (keeps perigee safe)
INC_MIN, INC_MAX = 0.0, math.pi               # full inclination range


def generate_random_params(
    n_samples,
    params_path="output/random_orbital_parameters.csv",
    states_path="output/initial_state_vectors.csv",
    seed=42,
):
    """
    Generate n_samples random LEO orbits and save two aligned CSVs.

      * params_path — Keplerian elements + fixed receiver/sim settings.
      * states_path — initial ECEF state vector [x, y, z, vx, vy, vz].

    The two files share sat_idx as their key (satellite i is row i in both).
    """
    rng = random.Random(seed)               # reproducible batch
    epoch = get_J2000_epoch_offset(EPOCH_STR)

    param_rows = []
    state_rows = []

    for i in range(n_samples):
        sma  = rng.uniform(SMA_MIN, SMA_MAX)
        ecc  = rng.uniform(ECC_MIN, ECC_MAX)
        inc  = rng.uniform(INC_MIN, INC_MAX)
        raan = rng.uniform(0.0, 2 * math.pi)
        argp = rng.uniform(0.0, 2 * math.pi)
        anom = rng.uniform(0.0, 2 * math.pi)

        # Elements -> inertial (ECI / EME2000) Cartesian state
        state_eci = elem_to_pv(
            Frame.EME2000, epoch, sma, ecc, inc, raan, argp, anom, PositionAngle.MEAN
        )

        # Inertial (ECI) -> Earth-fixed (ECEF / ITRF) at the same epoch
        state_ecef = transform_frame(Frame.EME2000, epoch, state_eci, ECEF_FRAME)
        x, y, z, vx, vy, vz = list(state_ecef)

        param_rows.append([
            i, sma, ecc, inc, raan, argp, anom,
            STATION_NAME, STATION_LAT, STATION_LON, STATION_ALT,
            DELTA_T, CADENCE_S, RANGE_RATE_ERROR_MS,
        ])
        state_rows.append([i, x, y, z, vx, vy, vz])

    out_dir = os.path.dirname(params_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- CSV 1: Keplerian parameters + settings ---
    with open(params_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sat_idx",
            "sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "anom_rad",
            "station_name", "station_lat_rad", "station_lon_rad", "station_alt_m",
            "delta_t_s", "cadence_s", "range_rate_error_ms",
        ])
        writer.writerows(param_rows)
    print(f"Saved {n_samples} parameter sets to {params_path}")

    # --- CSV 2: initial ECEF state vectors ---
    with open(states_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sat_idx",
            "x_ecef_m", "y_ecef_m", "z_ecef_m",
            "vx_ecef_ms", "vy_ecef_ms", "vz_ecef_ms",
        ])
        writer.writerows(state_rows)
    print(f"Saved {n_samples} ECEF state vectors to {states_path}")


if __name__ == "__main__":
    # Only a fraction of orbits produce a Maui pass, so generate generously to
    # end up with many non-empty passes. Check how many come back non-empty and
    # raise this if you need more.
    generate_random_params(n_samples=20000)