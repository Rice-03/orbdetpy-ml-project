# =============================================================================
# generate_params.py  -  Constructive LEO orbit + ECEF state vector generator
# =============================================================================
#
# PURPOSE
# -------
# Generate LEO Keplerian orbits that are CONSTRUCTED to produce a visible pass
# over a fixed equatorial ground station, for batch input into simulationv2.py,
# and convert each orbit into its initial state vector in the EARTH-FIXED
# (ECEF / ITRF) frame via orbdetpy.
#
# This replaces the old rejection sampler (draw random elements, propagate,
# keep whatever happens to clear the horizon -> most draws wasted). Instead,
# each satellite is placed directly at its CLOSEST APPROACH to the station,
# and a physically consistent velocity is built for it there, so ~100% of
# draws produce a strong pass, well-centred in the propagation window.
#
# Two CSVs are written, one row per satellite (aligned by sat_idx):
#
#   1. random_orbital_parameters.csv - six Keplerian elements (labels), the
#                                       seed-point geometry (diagnostics), and
#                                       fixed receiver + simulation settings.
#                                       (input for simulationv2.py)
#
#   2. initial_state_vectors.csv     - [x, y, z, vx, vy, vz] in the ECEF frame
#                                       at prop_start, one satellite per row.
#                                       (evidence trail, not model input)
#
# FRAME NOTE
# ----------
# The orbit is built directly in the INERTIAL frame (EME2000 / ECI): the look
# direction is worked out in ECEF (where the station is fixed), rotated once
# into ECI as a position only, and the velocity is constructed entirely in ECI
# from there. Velocity is never built in ECEF and then transformed, because
# that would inject the Earth-rotation term (omega x r) into it.
#
# CONSTRUCTION SUMMARY
# ---------------------
# For each satellite:
#   1. Sample sma, ecc (coupled - see ecc_max), a seed true anomaly nu, a
#      look direction (zenith angle z0 <= 50 deg, i.e. elevation >= 40 deg,
#      azimuth az) above the station, a seed time t_star (spread over one
#      sidereal day), and a pass-direction sign (one of the two geometries
#      solved for in step 4).
#   2. Place the satellite along that look direction, at the geocentric
#      radius implied by (sma, ecc, nu), by intersecting the look ray with
#      the satellite's radius sphere. Rotate that position into ECI.
#   3. Build the velocity so the satellite is at CLOSEST APPROACH to the
#      station at t_star: the flight-path angle and speed relative to the
#      geocentric radius are fixed by two-body dynamics (sma, ecc, nu) --
#      what's still free is which direction, within the plane perpendicular
#      to the geocentric radius, the tangential part of the velocity points.
#      That direction is solved for directly so the full velocity vector is
#      perpendicular to the station line of sight (zero closing/receding
#      range rate at t_star -- the definition of closest approach), rather
#      than chosen to hit a sampled inclination. Inclination falls out as a
#      RESULT of this, not an input; resample if no solution exists, or if
#      the resulting inclination falls outside the [5, 100] deg LEO band.
#   4. Convert the resulting (r, v) at the seed time back to classical
#      elements (a standalone rv_to_coe, cross-checked against orbdetpy's
#      pv_to_elem where available - see the smoke test), then back-propagate
#      the mean anomaly by half the search window (two-body) to get the
#      element set at prop_start, which is what gets written out.
#
# Building the seed point as closest approach (rather than an arbitrary point
# within the visibility cone) is what keeps the pass centred in the window:
# closest approach is very close to peak elevation for a near-circular orbit,
# which is close to the true temporal middle of the visible arc, so the
# window's fixed W/2 margin on each side is far less likely to run out
# before AOS/after LOS than it was with an arbitrarily-placed seed point.
#
# =============================================================================

import math
import csv
import os
import random

from orbdetpy import Frame, PositionAngle
from orbdetpy.conversion import get_J2000_epoch_offset, elem_to_pv, transform_frame, lla_to_pos

# pv_to_elem is used only to cross-check rv_to_coe's convention during the
# smoke test (Risk G1 - not confirmed present in every orbdetpy version).
# Correctness of the generator does NOT depend on it being available.
try:
    from orbdetpy.conversion import pv_to_elem
    _HAVE_PV_TO_ELEM = True
except ImportError:
    _HAVE_PV_TO_ELEM = False


# --- Fixed receiver location: moved to the equator (was Maui) ---
STATION_NAME = "Equator"
STATION_LAT  = 0.0      # [rad]
STATION_LON  = -2.7272  # [rad]  (unchanged)
STATION_ALT  = 0.0      # [m]

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

# Reference epoch string; J2000_BASE below is derived from it
EPOCH_STR = "2019-05-01T00:00:00"

# Frames
ECI  = Frame.EME2000
ECEF = Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP

# Physical constants
MU    = 3.986004418e14   # Earth GM [m^3/s^2]
R_E   = 6378137.0        # Earth equatorial radius [m] (station is at lat 0, alt 0)
T_SID = 86164.0905       # sidereal day [s]

# --- Construction geometry ---
SEARCH_WINDOW_S = 24 * 60   # W; the seed instant (closest approach) sits at W/2
ZENITH_MAX_DEG  = 50.0      # look direction stays within 50 deg of zenith (elevation >= 40 deg)

# --- Orbit regime bounds (kept as parameters so MEO can be added later) ---
PERIGEE_ALT_MIN = 330_000.0    # [m] Orekit integrator floor
APOGEE_ALT_MAX  = 2_000_000.0  # [m] LEO regime ceiling
SMA_MIN, SMA_MAX = R_E + 400_000.0, R_E + 1_500_000.0
ECC_HARD_MAX = 0.02
# Inclination is no longer sampled (see module docstring, step 3): it falls
# out of the closest-approach velocity solve. These bounds are now a filter
# on that RESULT -- a solution outside [5, 100] deg is treated the same as
# no solution and resampled, keeping the same "real LEO is prograde" cap.
INC_MIN_DEG, INC_MAX_DEG = 5.0, 100.0


# --- small vector helpers (plain lists, no numpy dependency here) ---
def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

def _norm(a):
    return math.sqrt(_dot(a, a))

def _scale(a, s):
    return [x * s for x in a]

def _add(a, b):
    return [x + y for x, y in zip(a, b)]

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def ecc_max(sma):
    """
    Section B: the eccentricity ceiling coupled to sma, so perigee never dips
    below the Orekit integrator floor and apogee never exceeds the LEO
    ceiling. e is then drawn as a single U[0, ecc_max(sma)] sample, so
    reproducibility at a fixed seed is preserved (sma and ecc are NOT
    independent - widening one narrows the other's identifiability).
    """
    e_perigee = 1.0 - (R_E + PERIGEE_ALT_MIN) / sma
    e_apogee  = (R_E + APOGEE_ALT_MAX) / sma - 1.0
    return max(0.0, min(e_perigee, e_apogee, ECC_HARD_MAX))


def rv_to_coe(r_vec, v_vec):
    """
    Standalone Cartesian (ECI) state -> classical osculating elements.
    Shipped so correctness does not depend on orbdetpy's pv_to_elem being
    present or on trusting its return convention (Risk G1) - this function
    is what the F.3 round-trip check in the smoke test verifies.

    Returns (a, e, i, raan, argp, nu, M), angles in radians, all wrapped
    to [0, 2*pi) except a (semi-major axis, metres) and e (eccentricity).
    """
    r = _norm(r_vec)
    v = _norm(v_vec)
    h_vec = _cross(r_vec, v_vec)
    h = _norm(h_vec)
    n_vec = _cross([0.0, 0.0, 1.0], h_vec)
    n = _norm(n_vec)
    e_vec = [((v*v - MU/r) * r_vec[k] - _dot(r_vec, v_vec) * v_vec[k]) / MU for k in range(3)]
    e = _norm(e_vec)
    energy = v*v/2 - MU/r
    a = -MU / (2 * energy)
    i = math.acos(_clip(h_vec[2] / h, -1.0, 1.0))

    if n < 1e-8:  # near-equatorial guard (rare: INC_MIN_DEG keeps inc >= 5 deg)
        raan = 0.0
        p_hat = [1.0, 0.0, 0.0]
    else:
        raan = math.atan2(n_vec[1], n_vec[0]) % (2 * math.pi)
        p_hat = _scale(n_vec, 1.0 / n)

    h_hat = _scale(h_vec, 1.0 / h)
    q_hat = _cross(h_hat, p_hat)
    u = math.atan2(_dot(r_vec, q_hat), _dot(r_vec, p_hat))  # argument of latitude

    if e < 1e-8:
        nu = u % (2 * math.pi)
        argp = 0.0
    else:
        e_hat = _scale(e_vec, 1.0 / e)
        q_e = _cross(h_hat, e_hat)
        nu = math.atan2(_dot(r_vec, q_e), _dot(r_vec, e_hat)) % (2 * math.pi)
        argp = (u - nu) % (2 * math.pi)

    E = 2 * math.atan2(
        math.sqrt(max(0.0, 1 - e)) * math.sin(nu / 2),
        math.sqrt(max(0.0, 1 + e)) * math.cos(nu / 2),
    )
    M = (E - e * math.sin(E)) % (2 * math.pi)
    return a, e, i, raan, argp, nu, M


def generate_random_params(
    n_samples,
    params_path="output/random_orbital_parameters.csv",
    states_path="output/initial_state_vectors.csv",
    seed=42,
):
    """
    Construct n_samples LEO orbits, each guaranteed (up to the feasibility
    check in step 3 of the module docstring) to produce a strong pass over
    the equatorial station, centred on its closest approach, and save two
    aligned CSVs (see module docstring). sat_idx is the shared key.
    """
    rng = random.Random(seed)
    J2000_BASE = get_J2000_epoch_offset(EPOCH_STR)
    W = SEARCH_WINDOW_S

    # Station ECEF position and local ENU basis: both time-invariant for a
    # fixed lat/lon/alt point, so computed once outside the loop.
    s_ecef = lla_to_pos(J2000_BASE, [STATION_LAT, STATION_LON, STATION_ALT])
    east  = [-math.sin(STATION_LON), math.cos(STATION_LON), 0.0]
    north = [0.0, 0.0, 1.0]
    up    = [math.cos(STATION_LON), math.sin(STATION_LON), 0.0]

    cos_zmax = math.cos(math.radians(ZENITH_MAX_DEG))
    inc_lo, inc_hi = math.radians(INC_MIN_DEG), math.radians(INC_MAX_DEG)

    param_rows = []
    state_rows = []
    total_retries_all = 0

    for idx in range(n_samples):
        n_retries = 0
        accepted = False

        while not accepted:
            # --- steps 1-2: sma, ecc (coupled ceiling, Section B) ---
            sma = rng.uniform(SMA_MIN, SMA_MAX)
            ecc = rng.uniform(0.0, ecc_max(sma))

            # --- remaining steps, with up to ~100 resamples on infeasibility ---
            for attempt in range(100):
                nu   = rng.uniform(0.0, 2 * math.pi)
                z0   = math.acos(rng.uniform(cos_zmax, 1.0))
                az   = rng.uniform(0.0, 2 * math.pi)
                tau  = rng.uniform(0.0, T_SID)
                t_star = J2000_BASE + tau
                sign_psi = rng.choice((-1.0, 1.0))  # which of the two closest-approach geometries

                # --- place the satellite along the look direction ---
                L = _add(_add(_scale(east, math.sin(z0) * math.sin(az)),
                              _scale(north, math.sin(z0) * math.cos(az))),
                         _scale(up, math.cos(z0)))
                r = sma * (1 - ecc**2) / (1 + ecc * math.cos(nu))
                b = _dot(s_ecef, L)
                disc = b*b - (R_E**2 - r**2)
                if disc < 0:
                    n_retries += 1
                    continue
                rho = -b + math.sqrt(disc)
                p_ecef = _add(s_ecef, _scale(L, rho))
                r_eci = transform_frame(ECEF, t_star, p_ecef + [0.0, 0.0, 0.0], ECI)[:3]
                L_eci = transform_frame(ECEF, t_star, L + [0.0, 0.0, 0.0], ECI)[:3]
                r_hat = _scale(r_eci, 1.0 / _norm(r_eci))
                delta = math.asin(_clip(r_hat[2], -1.0, 1.0))
                cos_delta = math.sqrt(max(1e-12, 1 - r_hat[2]**2))

                # --- closest-approach velocity direction solve ---
                # Flight-path angle and speed relative to r_hat are fixed by
                # two-body dynamics (sma, ecc, nu); the only remaining freedom
                # is psi, the direction of the tangential component within
                # the plane perpendicular to r_hat. Solve for the psi that
                # makes the full velocity vector perpendicular to L_eci (zero
                # range rate at t_star, i.e. closest approach to the station).
                w_hat = _scale([-r_hat[1], r_hat[0], 0.0], 1.0 / cos_delta)
                m_hat = _cross(r_hat, w_hat)
                gamma = math.atan2(ecc * math.sin(nu), 1 + ecc * math.cos(nu))

                A = _dot(w_hat, L_eci)
                B = _dot(m_hat, L_eci)
                C = -math.tan(gamma) * _dot(r_hat, L_eci)
                R_mag = math.sqrt(A*A + B*B)
                if R_mag < 1e-12 or abs(C) > R_mag:
                    n_retries += 1   # no psi solves v.L_eci = 0 for this geometry
                    continue
                phi0 = math.atan2(B, A)
                dpsi = math.acos(_clip(C / R_mag, -1.0, 1.0))
                psi = phi0 + sign_psi * dpsi

                t_hat = _add(_scale(w_hat, math.cos(psi)), _scale(m_hat, math.sin(psi)))
                V = math.sqrt(MU * (2.0 / r - 1.0 / sma))
                v_eci = _add(_scale(r_hat, V * math.sin(gamma)), _scale(t_hat, V * math.cos(gamma)))

                h_vec = _cross(r_eci, v_eci)
                h = _norm(h_vec)
                inc = math.acos(_clip(h_vec[2] / h, -1.0, 1.0))  # RESULT, not sampled

                if inc_lo <= inc <= inc_hi:
                    accepted = True
                    break
                n_retries += 1
            # if the for-loop exhausts without `accepted`, control falls back
            # to the top of the while-loop, which redraws sma AND ecc too.

        total_retries_all += n_retries

        # cheap correctness asserts - catch a sign/frame bug immediately
        assert abs(h - math.sqrt(MU * sma * (1 - ecc**2))) / h < 1e-6, "ang. momentum magnitude mismatch"
        assert abs(_dot(v_eci, L_eci)) / V < 1e-6, "velocity not perpendicular to line of sight at seed point"

        # --- state -> osculating elements at the seed time ---
        a_out, e_out, i_out, raan, argp, nu_out, M_tstar = rv_to_coe(r_eci, v_eci)

        # --- reference the elements to prop_start (two-body back-out) ---
        prop_start_i = t_star - W / 2.0
        epoch_offset_s = prop_start_i - J2000_BASE
        n_mm = math.sqrt(MU / a_out**3)
        M_prop_start = (M_tstar - n_mm * (W / 2.0)) % (2 * math.pi)

        # state at prop_start, for the evidence-trail CSV
        state_eci_prop = elem_to_pv(ECI, prop_start_i, a_out, e_out, i_out,
                                     raan, argp, M_prop_start, PositionAngle.MEAN)
        state_ecef_prop = transform_frame(ECI, prop_start_i, state_eci_prop, ECEF)
        x, y, z, vx, vy, vz = state_ecef_prop

        param_rows.append([
            idx, a_out, e_out, i_out, raan, argp, M_prop_start,
            STATION_NAME, STATION_LAT, STATION_LON, STATION_ALT,
            DELTA_T, CADENCE_S, RANGE_RATE_ERROR_MS,
            epoch_offset_s, W,
            nu, z0, delta, sign_psi, n_retries,
        ])
        state_rows.append([idx, x, y, z, vx, vy, vz, epoch_offset_s])

    out_dir = os.path.dirname(params_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(params_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sat_idx", "sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "anom_rad",
            "station_name", "station_lat_rad", "station_lon_rad", "station_alt_m",
            "delta_t_s", "cadence_s", "range_rate_error_ms",
            "epoch_offset_s", "search_window_s",
            "nu_seed_rad", "z0_seed_rad", "seed_geoc_lat_rad", "sign_psi", "n_retries",
        ])
        writer.writerows(param_rows)
    print(f"Saved {n_samples} parameter sets to {params_path}")

    with open(states_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sat_idx",
            "x_ecef_m", "y_ecef_m", "z_ecef_m",
            "vx_ecef_ms", "vy_ecef_ms", "vz_ecef_ms",
            "epoch_offset_s",
        ])
        writer.writerows(state_rows)
    print(f"Saved {n_samples} ECEF state vectors to {states_path}")

    print(f"Total retries consumed across batch: {total_retries_all}")
    if not _HAVE_PV_TO_ELEM:
        print("NOTE: orbdetpy.conversion.pv_to_elem not found in this install "
              "(Risk G1) - rv_to_coe is unaffected, it does not depend on it.")


if __name__ == "__main__":
    # Smoke-test scale first (per the plan): confirm this works before the
    # full 10,000-orbit batch.
    generate_random_params(n_samples=500)