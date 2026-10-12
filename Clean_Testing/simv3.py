# =============================================================================
# simulationv2.py  –  Doppler-difference simulation for LEO satellite overpasses
# =============================================================================
#
# PURPOSE
# -------
# This module provides simulate_doppler_pass(), a function that simulates the
# Doppler-proxy observable produced by a ground station during a LEO satellite
# overpass.  It computes the *difference* in delay between two successive probe
# signals transmitted delta_t seconds apart — the quantity a real receiver can
# actually measure without a shared clock with the satellite.
#
# PHYSICAL BACKGROUND
# -------------------
# A ground station cannot directly observe the absolute one-way propagation
# delay d(t) = range(t) / c, because that would require the station and
# satellite to share a common time reference.  What the station CAN measure is
# the difference in *arrival* time between two pulses transmitted exactly
# delta_t seconds apart:
#
#     Δd(t)  =  d(t + delta_t) − d(t)
#             =  [range(t + delta_t) − range(t)] / c          [seconds]
#
# This quantity is the discrete-time analogue of the Doppler range-rate:
#
#     Δd(t)  ≈  (d range/dt) · delta_t / c  =  range_rate(t) · delta_t / c
#
# Sign convention: Δd < 0 when the satellite is approaching (range shrinking),
# Δd > 0 when it is receding.  Δd passes through zero near the time of closest
# approach (TCA), which corresponds to zero Doppler shift.
#
# IMPLEMENTATION
# --------------
# For each satellite the function runs ONE propagation using orbdetpy with
# ONLY MeasurementType.RANGE_RATE configured — identical to the measurement
# configuration already verified.  Orekit computes range rate analytically
# from the orbit at each visible timestep, so no differencing of two
# nearly-equal large numbers is involved.
#
#     delay_diff[i]  =  range_rate[i] · delta_t / c          [seconds]
#
# PREVIOUS METHOD (removed — it was incorrect)
# --------------------------------------------
# Earlier versions ran two propagations, the second starting delta_t later but
# re-using the SAME initial state, then differenced the RANGE measurements by
# index.  Because propagation B started from the epoch state at time
# epoch+delta_t, the satellite had travelled the same elapsed time at each
# paired index — so it sat at an identical position in both runs.  The only
# thing that actually differed was the ground station, which had rotated with
# the Earth for 1 ms (~0.44 m).  The result therefore measured Earth rotation
# plus noise, not satellite Doppler.
#
# -----------------------------------------------------------------------------
# CHANGES IN V3 (this file, simv3.py) — per orbit-generation-plan.md Section C
# -----------------------------------------------------------------------------
# generate_params.py (V3) now constructs each orbit around a guaranteed pass at
# an EQUATORIAL station, with the pass seed placed at the middle of a per-
# satellite propagation window, and writes epoch_offset_s / search_window_s
# per row instead of everyone sharing one fixed epoch and an 80-minute window.
# This file is updated to match:
#
#   - epoch_str -> epoch_base_str: the fixed calendar reference (still
#     "2019-05-01T00:00:00", must match generate_params.py's EPOCH_STR).
#   - NEW epoch_offset_vec parameter: per-satellite offset [s] from
#     epoch_base_str. prop_start_i = epoch_base + epoch_offset_i, so each
#     satellite propagates over ITS OWN window, seeded around its own pass,
#     not a single shared epoch. None -> all zeros (back-compat with V1/V2
#     style input where every satellite shared one epoch).
#   - search_window_s default 80*60 -> 24*60: a strong LEO pass (peak
#     elevation > 30 deg) lasts at most ~10-12 minutes horizon to horizon;
#     1440 s gives ~2x margin, ~3x faster propagation, and less J2 drift
#     pulling the pass off-centre than the old 80-minute window did.
#   - times_s is now measured from each satellite's OWN prop_start_i, not a
#     shared epoch, so t_aos_s / t_max_elevation_s land near
#     search_window_s / 2 for every satellite instead of being scattered
#     across an 80-minute span most of which had no visible pass.
#
# LATER V3 UPDATE (generate_params.py only, no change needed in this file):
# the seed point at the window's middle is now constructed as the satellite's
# CLOSEST APPROACH to the station (velocity perpendicular to the line of
# sight, zero range rate), not an arbitrary point within the visibility cone,
# and the minimum elevation at that seed point was tightened from 30 to 40
# deg. Both changes live entirely in generate_params.py's geometry
# construction. This file still only reads six elements + epoch_offset_s +
# search_window_s per row -- unaffected either way.
#
# LATER V3 UPDATE #2 (this file): prop_start_i now gets a small per-satellite
# random jitter (see simulate_doppler_pass's own docstring and the
# max_jitter_s / jitter_seed parameters) instead of sitting at the exact,
# fixed epoch_base + epoch_offset_i for every row. Without it, every
# satellite's samples fall at the exact same phase relative to its own seed
# point (closest approach) -- a fixed pattern a model could learn to exploit
# instead of the physical curve shape. This is a sampling-time detail, not
# part of the orbit's ground truth, so it is not read from or written to the
# params CSV.
#
# Everything else -- the RANGE_RATE-only measurement configuration, the
# derived-elevation method, the per-satellite try/except, contiguity
# semantics, output CSV schemas, and the plot -- is UNCHANGED from v2.1.
# -----------------------------------------------------------------------------
# CHANGES IN THIS REVISION (v2.1) — VERIFIED AGAINST A LIVE ORBDETPY INSTALL
# -----------------------------------------------------------------------------
# Two pieces of information were previously discarded when writing the CSV:
# measurement timestamps, and pass geometry (elevation). Both are now captured.
#
# WHY ELEVATION IS NOT A CONFIGURED MEASUREMENT TYPE
# ---------------------------------------------------
# The first attempt at this feature enabled MeasurementType.ELEVATION (and
# AZIMUTH) alongside RANGE_RATE, on the assumption that orbdetpy would return
# all three values together in m.values, indexed by enum order. This was
# tested directly against a live orbdetpy install and found to be wrong in a
# way no amount of re-indexing could fix:
#
#   RANGE_RATE only:            m.values = [range_rate]                (len 1)
#   ELEVATION only:              m.values = [azimuth, elevation]        (len 2)
#   AZIMUTH only:                 m.values = [azimuth, elevation]        (len 2)
#   ELEVATION + RANGE_RATE:      m.values = [azimuth, elevation]        (len 2)
#   AZIMUTH + ELEVATION + RANGE_RATE: m.values = [azimuth, elevation]   (len 2)
#   RANGE + RANGE_RATE:           m.values = [range, range_rate]         (len 2)
#
# Enabling any angle measurement (AZIMUTH or ELEVATION) completely SUPPRESSES
# RANGE_RATE from the output, regardless of what else is configured alongside
# it. This is a real constraint in how orbdetpy groups simulated measurement
# types, not an indexing mistake — RANGE_RATE and the angle pair cannot be
# retrieved from the same propagation call together.
#
# THE FIX: DERIVE ELEVATION FROM true_state INSTEAD
# ---------------------------------------------------
# Every measurement object carries m.true_state — the satellite's true
# position/velocity (EME2000/ECI frame) at that timestep — regardless of which
# measurement type is configured. This was confirmed present even with
# RANGE_RATE as the only configured type.
#
# So the measurement configuration stays EXACTLY as already verified
# (RANGE_RATE only). Elevation and azimuth are computed afterward, in pure
# Python, from m.true_state:
#   1. Transform true_state from EME2000 (ECI) to ECEF using transform_frame,
#      at the measurement's own time (not the reference epoch — Earth rotates).
#   2. Get the station's fixed ECEF position via lla_to_pos.
#   3. Rotate the satellite-minus-station vector into the station's local
#      East-North-Up frame; elevation = asin(Up / range).
#
# VALIDATION
# ----------
# This geometric elevation was checked against orbdetpy's own ELEVATION
# measurement (from a separate, throwaway propagation) across a full real pass
# from this project's actual seed=42 batch (satellite 0: sma=7,289,541 m,
# ecc=0.0003, inc=49.51 deg). Agreement was within 0.02-0.12 degrees, growing
# larger near the horizon and smaller near the elevation peak — the signature
# of atmospheric refraction, which orbdetpy's ELEVATION model apparently
# includes and this vacuum-geometry calculation does not. At LEO range this is
# a few hundred metres of cross-range error at most, negligible for a pass-
# geometry feature. The visibility mask (which timesteps get a measurement)
# was confirmed identical between the RANGE_RATE-only and ELEVATION-only runs.
#
# NOTE ON FEATURE vs LABEL
# ------------------------
# pass_geometry.csv is TRUTH data derived from the known orbit — it is a LABEL
# file (like initial_state_vectors.csv), not a feature file. It must not be fed
# to the model as input; at inference only the delay sequence is observable.
#
# CONTIGUITY CHECK
# -----------------
# t_aos_s + index*cadence only reconstructs every timestamp if the visible
# samples are consecutive cadence steps. This is checked per satellite. Note
# that Orekit clips the LAST sample of a pass to the exact search-window
# boundary rather than the next full cadence step, so a pass still in progress
# when the window ends legitimately has a SHORTER final interval — this is not
# a gap and is explicitly allowed. Only an interval LARGER than the cadence
# (satellite dropping below the horizon mid-pass and reappearing) counts as a
# genuine gap.
#
# INPUTS
# ------
# sma_vec, ecc_vec, inc_vec, raan_vec, argp_vec, anom_vec : sequences of N
#     Keplerian elements, one satellite's orbit per index (EME2000 frame).
#
# OUTPUT
# ------
# results : list of length N of dicts, one per satellite:
#     {
#       "times_s"       : [float]  seconds after the reference epoch
#       "range_rates"   : [float]  line-of-sight range rate [m/s]
#       "elevations"    : [float]  elevation above horizon [rad], derived
#       "azimuths"      : [float]  azimuth [rad], derived
#       "delays"        : [float]  delay-difference observable [s]
#       "contiguous"    : bool     True if samples are consecutive cadence steps
#     }
#     All lists are empty (and contiguous is True) if no overpass was detected.
# =============================================================================

from orbdetpy import (configure, add_station, MeasurementType, Frame, PositionAngle)
from orbdetpy.conversion import (get_J2000_epoch_offset, get_UTC_string, elem_to_pv,
                                  lla_to_pos, transform_frame)
from orbdetpy.propagation import propagate_orbits
import matplotlib.pyplot as plt
import math
import csv
import os
import random
import time


ECEF_FRAME = Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP


def _station_ecef(epoch, lat, lon, alt):
    """Fixed station position in ECEF [m], via WGS-84 lat/lon/alt."""
    return list(lla_to_pos(epoch, [lat, lon, alt]))


def _station_enu_basis(lat, lon):
    """East/North/Up unit vectors (in ECEF) at the station's location."""
    east  = [-math.sin(lon), math.cos(lon), 0.0]
    north = [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)]
    up    = [ math.cos(lat) * math.cos(lon),  math.cos(lat) * math.sin(lon), math.sin(lat)]
    return east, north, up


def _elevation_azimuth_from_true_state(true_state, m_time, station_ecef, east, north, up):
    """
    Derive elevation and azimuth from the satellite's true ECI state at a
    given measurement time, without configuring ELEVATION/AZIMUTH on the
    propagation (see module docstring for why). Vacuum geometry — no
    atmospheric refraction correction (negligible at LEO range for this use).
    """
    sat_ecef = list(transform_frame(Frame.EME2000, m_time, list(true_state), ECEF_FRAME))[:3]
    dx = [sat_ecef[i] - station_ecef[i] for i in range(3)]
    rng = math.sqrt(sum(v * v for v in dx))
    e = sum(dx[i] * east[i] for i in range(3))
    n = sum(dx[i] * north[i] for i in range(3))
    u = sum(dx[i] * up[i] for i in range(3))
    elevation = math.asin(max(-1.0, min(1.0, u / rng)))
    azimuth = math.atan2(e, n) % (2 * math.pi)
    return elevation, azimuth


def simulate_doppler_pass(
    sma_vec, ecc_vec, inc_vec, raan_vec, argp_vec, anom_vec,
    epoch_offset_vec=None,
    epoch_base_str="2019-05-01T00:00:00",
    station_name="Maui",
    station_lat=0.3614,
    station_lon=-2.7272,
    station_alt=3059.0,
    rso_mass=2500.0,
    range_rate_error_ms=0.0,
    cadence_s=30.0,
    search_window_s=24 * 60,   # 24 min: ~2x margin over a strong LEO pass (<=~10-12 min); see V3 note above
    delta_t=0.001,
    max_jitter_s=None,   # None -> cadence_s / 2 (see note below)
    jitter_seed=1234,    # separate from generate_params.py's seed -- this is a sampling-time detail, not part of the orbit's ground truth
):
    """
    Simulate the Doppler observable and pass geometry for each satellite over
    one overpass. Each satellite propagates over its own window,
    prop_start_i = epoch_base + epoch_offset_i, so a batch of satellites
    built around different seed times (generate_params.py V3) is each
    referenced to its own pass rather than one shared epoch. See module
    docstring for full method notes.

    prop_start_i also gets a small per-satellite random jitter (uniform in
    [-max_jitter_s, +max_jitter_s], default half a cadence step) so sampling
    does not begin at a fixed, exact offset from the seed point (closest
    approach) for every satellite -- otherwise every row would have its
    samples falling on the exact same phase relative to t_star, a pattern a
    model could latch onto instead of the physical curve shape. The window
    length (search_window_s) is unchanged; jitter shifts both edges together,
    so it costs at most max_jitter_s off the AOS/LOS margins established by
    the closest-approach construction -- small by construction, not a fix
    for the margin itself.
    """
    c = 299792458.0
    epoch_base = get_J2000_epoch_offset(epoch_base_str)
    station_ecef = _station_ecef(epoch_base, station_lat, station_lon, station_alt)
    east, north, up = _station_enu_basis(station_lat, station_lon)
    results = []

    if epoch_offset_vec is None:
        epoch_offset_vec = [0.0] * len(sma_vec)  # back-compat: everyone shares epoch_base

    if max_jitter_s is None:
        max_jitter_s = cadence_s / 2.0
    jitter_rng = random.Random(jitter_seed)

    for sma, ecc, inc, raan, argp, anom, epoch_offset in zip(
        sma_vec, ecc_vec, inc_vec, raan_vec, argp_vec, anom_vec, epoch_offset_vec
    ):
        jitter = jitter_rng.uniform(-max_jitter_s, max_jitter_s)
        prop_start_i = epoch_base + epoch_offset + jitter

        initial_state = elem_to_pv(
            Frame.EME2000, prop_start_i, sma, ecc, inc, raan, argp, anom, PositionAngle.MEAN
        )

        # Any single orbit that Orekit cannot propagate (e.g. a pathological
        # low-perigee draw) is caught here and recorded as an empty pass, so
        # one bad satellite never aborts the whole batch.
        try:
            cfg = configure(
                rso_mass=rso_mass,
                prop_start=prop_start_i,
                prop_initial_state=initial_state,
                prop_end=prop_start_i + search_window_s,
                prop_step=cadence_s,
                sim_measurements=True,
            )
            add_station(cfg, station_name, station_lat, station_lon, station_alt)

            # Configuration is UNCHANGED from the version already verified —
            # RANGE_RATE only. Elevation/azimuth are derived below, not
            # configured here (see module docstring for why).
            cfg.measurements[MeasurementType.RANGE_RATE].error[:] = [range_rate_error_ms]

            meas = list(propagate_orbits([cfg])[0].array)

            times_s     = [m.time - prop_start_i for m in meas]
            range_rates = [m.values[0] for m in meas]
            delays      = [rr * delta_t / c for rr in range_rates]

            elevations, azimuths = [], []
            for m in meas:
                el, az = _elevation_azimuth_from_true_state(
                    m.true_state, m.time, station_ecef, east, north, up
                )
                elevations.append(el)
                azimuths.append(az)

            # See module docstring: a shorter-than-cadence FINAL interval is
            # expected (window-boundary clipping), not a gap. Only a LARGER
            # interval is a genuine gap.
            contiguous = True
            for k, (t_prev, t_next) in enumerate(zip(times_s, times_s[1:])):
                interval = t_next - t_prev
                is_last = (k == len(times_s) - 2)
                if interval > cadence_s + 0.5:
                    contiguous = False
                    break
                if interval < cadence_s - 0.5 and not is_last:
                    contiguous = False
                    break

        except Exception as exc:
            print(f"  [warn] propagation failed (sma={sma:.0f}, ecc={ecc:.4f}, "
                  f"inc={inc:.3f}): {type(exc).__name__}; recording empty pass.")
            times_s, azimuths, elevations, range_rates, delays = [], [], [], [], []
            contiguous = True

        results.append({
            "times_s":     times_s,
            "azimuths":    azimuths,
            "range_rates": range_rates,
            "elevations":  elevations,
            "delays":      delays,
            "contiguous":  contiguous,
        })

    return results


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":

    # Set to a small number (e.g. 20) to smoke-test the run before committing
    # to the full ~2.4 hr batch. None = process every satellite in the file.
    TEST_LIMIT = None

    params_in_path = "output/random_orbital_parameters.csv"
    if not os.path.exists(params_in_path):
        raise SystemExit(
            f"Parameter file not found: {params_in_path}\n"
            "Run generate_params.py first to create it."
        )

    sma_vec, ecc_vec, inc_vec = [], [], []
    raan_vec, argp_vec, anom_vec = [], [], []
    epoch_offset_vec = []
    with open(params_in_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if TEST_LIMIT is not None:
            rows = rows[:TEST_LIMIT]
        for row in rows:
            sma_vec.append(float(row["sma_m"]))
            ecc_vec.append(float(row["ecc"]))
            inc_vec.append(float(row["inc_rad"]))
            raan_vec.append(float(row["raan_rad"]))
            argp_vec.append(float(row["argp_rad"]))
            anom_vec.append(float(row["anom_rad"]))
            epoch_offset_vec.append(float(row["epoch_offset_s"]))

    if not rows:
        raise SystemExit(f"No satellites found in {params_in_path}.")

    # Station and timing settings are identical across rows (the generator
    # writes them as constants), so take them from the first row.
    first = rows[0]
    STATION_NAME = first["station_name"]
    STATION_LAT  = float(first["station_lat_rad"])
    STATION_LON  = float(first["station_lon_rad"])
    STATION_ALT  = float(first["station_alt_m"])
    DELTA_T      = float(first["delta_t_s"])
    CADENCE_S    = float(first["cadence_s"])
    RANGE_RATE_ERROR_MS = float(first["range_rate_error_ms"])
    SEARCH_WINDOW_S = float(first["search_window_s"])   # was hardcoded 80*60; now per-batch from the file

    # Fixed calendar reference epoch_offset_s is measured from. NOT in the
    # CSV -- must match generate_params.py's EPOCH_STR by convention.
    EPOCH_BASE_STR = "2019-05-01T00:00:00"

    c_light = 299792458.0

    n_sats_in = len(sma_vec)
    print(f"Loaded {len(rows)} satellites from {params_in_path}")
    if TEST_LIMIT is not None:
        print(f"  [TEST_LIMIT active: only the first {TEST_LIMIT} will be run]")

    print(f"\n=== SIMULATION STARTED === ({n_sats_in} satellites)")
    t_start = time.time()

    results = simulate_doppler_pass(
        sma_vec, ecc_vec, inc_vec, raan_vec, argp_vec, anom_vec,
        epoch_offset_vec=epoch_offset_vec,
        epoch_base_str=EPOCH_BASE_STR,
        station_name=STATION_NAME,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        station_alt=STATION_ALT,
        range_rate_error_ms=RANGE_RATE_ERROR_MS,
        cadence_s=CADENCE_S,
        search_window_s=SEARCH_WINDOW_S,
        delta_t=DELTA_T,
    )

    t_elapsed = time.time() - t_start
    print(f"=== SIMULATION FINISHED === "
          f"{t_elapsed:.1f} s ({t_elapsed/60:.2f} min, {t_elapsed/3600:.2f} hr)\n")

    os.makedirs("output", exist_ok=True)

    n_pass = sum(1 for r in results if r["delays"])
    print(f"{n_pass} of {len(results)} satellites produced a visible pass.")

    # --- Physics sanity check: range-rate band ---
    all_rr = [v for r in results for v in r["range_rates"]]
    if all_rr:
        print(f"Range rate: min {min(all_rr):>9.1f} m/s   max {max(all_rr):>9.1f} m/s")
        if max(abs(min(all_rr)), abs(max(all_rr))) > 8000.0:
            print("  [WARN] range rate outside realistic LEO band -- check the observable.")
        else:
            print("  OK: within the realistic LEO band.")

    # --- Physics sanity check: elevation band ---
    all_el = [v for r in results for v in r["elevations"]]
    if all_el:
        print(f"Elevation:  min {math.degrees(min(all_el)):>9.2f} deg  "
              f"max {math.degrees(max(all_el)):>9.2f} deg")
        if min(all_el) < -0.05 or max(all_el) > (math.pi / 2 + 0.05):
            print("  [WARN] elevation outside [0, 90] deg -- geometry calculation is suspect.")
        else:
            print("  OK: within the horizon-to-zenith band.")

    # --- Zero-crossing check: a real Doppler pass usually crosses zero ---
    n_cross = 0
    for r in results:
        d = r["delays"]
        if len(d) > 1 and any(a * b < 0 for a, b in zip(d, d[1:])):
            n_cross += 1
    print(f"Zero crossing: {n_cross} of {n_pass} passes cross zero "
          f"(approach -> recession).")

    # --- Timing sanity check: contiguity ---
    non_contig = [i for i, r in enumerate(results) if r["delays"] and not r["contiguous"]]
    print(f"Contiguity: {len(non_contig)} of {n_pass} non-empty passes have gaps.")
    if non_contig:
        print(f"  [WARN] non-contiguous sat_idx (first 10): {non_contig[:10]}")
    else:
        print("  OK: every pass is a consecutive run of cadence steps.")

    # --- AOS spread: confirms passes really are scattered through the window ---
    all_aos = [r["times_s"][0] for r in results if r["times_s"]]
    if all_aos:
        print(f"AOS time:   min {min(all_aos):>9.1f} s    max {max(all_aos):>9.1f} s "
              f"(window is {SEARCH_WINDOW_S} s)")

    # Per-satellite table print is OFF by default -- for a large batch this
    # can be hundreds of thousands of lines. Set VERBOSE_PER_SAT=True for
    # small test batches (e.g. 10-50 satellites).
    VERBOSE_PER_SAT = False
    if VERBOSE_PER_SAT:
        for sat_idx, r in enumerate(results):
            if not r["delays"]:
                print(f"\nSatellite {sat_idx}: no visible pass detected.")
                continue
            n = len(r["delays"])
            print(f"\nSatellite {sat_idx} — {n} samples, AOS at {r['times_s'][0]:.1f} s")
            print("-" * 96)
            print(f"{'t since epoch (s)':>18} {'Range rate (m/s)':>18} "
                  f"{'Delay diff (s)':>22} {'Elev (deg)':>14}")
            print("-" * 96)
            for t, rr, dd, el in zip(r["times_s"], r["range_rates"],
                                     r["delays"], r["elevations"]):
                print(f"{t:>18.1f} {rr:>18.1f} {dd:>22.12f} {math.degrees(el):>14.3f}")
            print("-" * 96)

    # NOTE: the orbital parameters are NOT re-written here. They already live
    # in output/random_orbital_parameters.csv (written by generate_params.py),
    # and this run reads from that file. Both CSVs below align to it by
    # sat_idx / row order, forming the labelled dataset.

    max_len = max((len(r["delays"]) for r in results), default=0)

    # --- CSV 1: delay differences (FEATURES) ---------------------------------
    data_path = "output/delay_differences.csv"
    with open(data_path, "w", newline="") as f:
        writer = csv.writer(f)
        dd_cols = [f"dd_{i:03d}" for i in range(max_len)]
        writer.writerow(["sat_idx", "num_valid_samples", "t_aos_s", "contiguous"] + dd_cols)
        for sat_idx, r in enumerate(results):
            delays = r["delays"]
            n = len(delays)
            t_aos = r["times_s"][0] if n else ""
            padded = delays + [0.0] * (max_len - n)
            writer.writerow([sat_idx, n, t_aos, int(r["contiguous"])] + padded)
    print(f"\nDelay-difference data saved to {data_path}")

    # --- CSV 2: pass geometry (LABELS) ---------------------------------------
    # Truth data derived from the known orbit. This is a LABEL file, like
    # initial_state_vectors.csv -- it must NOT be used as model input, since at
    # inference only the delay sequence is observable.
    geom_path = "output/pass_geometry.csv"
    with open(geom_path, "w", newline="") as f:
        writer = csv.writer(f)
        el_cols = [f"el_{i:03d}" for i in range(max_len)]
        writer.writerow([
            "sat_idx", "num_valid_samples", "t_aos_s",
            "max_elevation_deg", "t_max_elevation_s",
            "elev_first_deg", "elev_last_deg",
            "az_first_deg", "az_last_deg",
        ] + el_cols)
        for sat_idx, r in enumerate(results):
            el = r["elevations"]
            az = r["azimuths"]
            n = len(el)
            if n:
                k_max = max(range(n), key=lambda k: el[k])
                t_aos = r["times_s"][0]
                row_head = [
                    sat_idx, n, t_aos,
                    math.degrees(el[k_max]),
                    r["times_s"][k_max] - t_aos,
                    math.degrees(el[0]),
                    math.degrees(el[-1]),
                    math.degrees(az[0]),
                    math.degrees(az[-1]),
                ]
            else:
                row_head = [sat_idx, 0, "", "", "", "", "", "", ""]
            padded_el = [math.degrees(e) for e in el] + [0.0] * (max_len - n)
            writer.writerow(row_head + padded_el)
    print(f"Pass geometry saved to {geom_path}")

    # --- Combined plot ---------------------------------------------------
    PLOT_LIMIT = 12
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for sat_idx, r in enumerate(results):
        if not r["delays"]:
            continue
        if plotted >= PLOT_LIMIT:
            break
        t_aos = r["times_s"][0]
        times_rel = [t - t_aos for t in r["times_s"]]
        ax.plot(times_rel, r["delays"], marker="o",
                color=colors[plotted % len(colors)],
                label=f"Satellite {sat_idx}")
        plotted += 1
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time since AOS (s)")
    ax.set_ylabel("Delay difference Δd (s)")
    ax.set_title(f"Doppler delay difference — first {plotted} passes (δt = {DELTA_T} s)")
    ax.legend(fontsize="small")
    ax.grid(True)
    plt.tight_layout()
    plot_path = "output/doppler_all_satellites.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()