# =============================================================================
# build_dataset.py  –  Turn raw Doppler curves into a model-ready training table
# =============================================================================
#
# PURPOSE
# -------
# delay_differences.csv has one row per satellite, most of them empty (no pass
# detected), and the real ones zero-padded to a uniform width — column dd_022
# means something different for every row depending on how long that pass was.
# A tree model (LightGBM, Random Forest) can't learn from that directly.
#
# This script:
#   1. CLEANS  — drops satellites with no pass, passes too short to describe,
#      and passes flagged non-contiguous (their timing math can't be trusted).
#   2. SUMMARISES — trims each remaining row to its real samples and computes
#      a fixed set of features (amplitude, timing, slope, curvature, shape).
#      Same columns for every satellite, regardless of how many raw samples
#      it had.
#   3. LABELS — joins in the ground truth: all six classical Keplerian
#      elements (sma, ecc, inc, raan, argp, anom) from
#      random_orbital_parameters.csv, by sat_idx. sma/ecc/inc are the
#      recoverable set; raan/argp/anom are included as a negative control
#      (tested, consistently negative R2 -- see project log / v3 proposal).
#
# Output is ONE table: feat_* columns (model input) and label_* columns
# (regression targets), never to be mixed. This is for the LightGBM / Random
# Forest path. A 1D CNN needs a different shape (padded sequence + mask) and
# is not built here.
#
# WHY THESE PARTICULAR FEATURES
# ------------------------------
# Each group targets something physically meaningful about the pass:
#   amplitude  -> how directly overhead the pass was (bigger swing = higher
#                 elevation), correlates with pass geometry
#   timing     -> when closest approach happened, how long the satellite was
#                 visible
#   slope      -> how fast the geometry was changing (steeper near a
#                 near-overhead pass)
#   curvature  -> second-order shape of the curve — this is the feature family
#                 most sensitive to noise (differencing amplifies it), worth
#                 watching once noise is reintroduced
#   shape      -> total range swept during the pass (integral of range rate)
#
# UNITS
# -----
# Delay-difference values are converted back to range rate [m/s] before any
# feature is computed (range_rate = dd * c / delta_t) — these are far more
# physically interpretable than raw delay-difference seconds.
# =============================================================================

import csv
import math
import os

import numpy as np
import pandas as pd


C_LIGHT = 299792458.0

# A pass shorter than this has no reliable slope/curvature (curvature alone
# needs >=3 points) and is dropped rather than filled with placeholder values.
MIN_SAMPLES = 5


def _load_delay_differences(path):
    df = pd.read_csv(path)
    dd_cols = [c for c in df.columns if c.startswith("dd_")]
    return df, dd_cols


def _load_labels(params_path):
    """
    Ground-truth labels, keyed by sat_idx. All six classical Keplerian
    elements, read directly from random_orbital_parameters.csv -- no new
    simulation needed, they were already being generated for every satellite,
    just not previously exposed as labels.

    sma, ecc, inc are the recoverable set (Phase 5/9/10 evidence).
    raan, argp, anom are included deliberately as a negative control: tested
    on identical features/model/split, they return negative R2, which is the
    direct evidence that phase and orientation are not recoverable from a
    single partial arc -- not an assumption, a measured result sitting
    alongside the parameters that DO work.

    Pass geometry (elevation/azimuth/max_elevation, from pass_geometry.csv)
    is NOT loaded here. It was tested as a label candidate and dropped: none
    of it is a true orbital parameter, and the one candidate with real signal
    (azimuth at AOS) was too weak to justify inclusion. pass_geometry.csv is
    still written by simulationv2.py (documents the calculation described in
    report section 3.3.2) but is no longer read by this script.
    """
    params = pd.read_csv(params_path)
    labels = params[["sat_idx", "sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "anom_rad"]].copy()
    labels["inc_deg"]  = np.degrees(labels["inc_rad"])
    labels["raan_deg"] = np.degrees(labels["raan_rad"])
    labels["argp_deg"] = np.degrees(labels["argp_rad"])
    labels["anom_deg"] = np.degrees(labels["anom_rad"])
    labels = labels.drop(columns=["inc_rad", "raan_rad", "argp_rad", "anom_rad"])

    labels = labels.rename(columns={
        "sma_m": "label_sma_m",
        "ecc": "label_ecc",
        "inc_deg": "label_inc_deg",
        "raan_deg": "label_raan_deg",
        "argp_deg": "label_argp_deg",
        "anom_deg": "label_anom_deg",
    })
    return labels


def extract_features(range_rate, cadence_s):
    """
    range_rate : 1D array of range-rate values [m/s], already trimmed to the
                 satellite's real samples (no padding), assumed evenly spaced
                 at cadence_s (only true for contiguous passes -- callers
                 should filter non-contiguous rows out before calling this).

    Returns a dict of feature_name -> value.
    """
    rr = np.asarray(range_rate, dtype=float)
    n = len(rr)

    # --- amplitude group ---
    dd_min, dd_max = float(rr.min()), float(rr.max())
    peak_to_peak = dd_max - dd_min
    dd_mean, dd_std = float(rr.mean()), float(rr.std())
    dd_first, dd_last = float(rr[0]), float(rr[-1])

    # --- timing group ---
    duration_s = (n - 1) * cadence_s

    # --- slope group --- (first derivative, m/s^2)
    d1 = np.diff(rr) / cadence_s
    slope_mean, slope_max, slope_min = float(d1.mean()), float(d1.max()), float(d1.min())

    # --- curvature group --- (second derivative, needs >=3 samples)
    if n >= 3:
        d2 = np.diff(rr, 2) / (cadence_s ** 2)
        curvature_mean, curvature_max = float(d2.mean()), float(d2.max())
    else:
        curvature_mean, curvature_max = np.nan, np.nan

    # --- shape group --- (total range swept during the pass, metres)
    integral_m = float(np.trapezoid(rr, dx=cadence_s)) if n > 1 else 0.0

    return {
        "feat_n_samples":            n,
        "feat_duration_s":           duration_s,
        "feat_dd_min":               dd_min,
        "feat_dd_max":               dd_max,
        "feat_peak_to_peak":         peak_to_peak,
        "feat_dd_mean":              dd_mean,
        "feat_dd_std":               dd_std,
        "feat_dd_first":             dd_first,
        "feat_dd_last":              dd_last,
        "feat_slope_mean":           slope_mean,
        "feat_slope_max":            slope_max,
        "feat_slope_min":            slope_min,
        "feat_curvature_mean":       curvature_mean,
        "feat_curvature_max":        curvature_max,
        "feat_integral_m":           integral_m,
    }


def build_training_table(
    delay_path="output/delay_differences.csv",
    params_path="output/random_orbital_parameters.csv",
    min_samples=MIN_SAMPLES,
):
    dd_df, dd_cols = _load_delay_differences(delay_path)
    labels = _load_labels(params_path)

    n_total = len(dd_df)
    n_no_pass = int((dd_df["num_valid_samples"] == 0).sum())

    usable = dd_df[dd_df["num_valid_samples"] > 0].copy()
    n_too_short = int((usable["num_valid_samples"] < min_samples).sum())
    usable = usable[usable["num_valid_samples"] >= min_samples]

    n_non_contig = int((usable["contiguous"] == 0).sum())
    usable = usable[usable["contiguous"] == 1]

    # Need delta_t_s (constant per satellite in generate_params.py, but read
    # per-row rather than hardcode, in case that ever changes) and cadence_s.
    params_lookup = pd.read_csv(params_path).set_index("sat_idx")[["delta_t_s", "cadence_s"]]

    feature_rows = []
    for _, row in usable.iterrows():
        sat_idx = int(row["sat_idx"])
        n = int(row["num_valid_samples"])
        delta_t_s = float(params_lookup.loc[sat_idx, "delta_t_s"])
        cadence_s = float(params_lookup.loc[sat_idx, "cadence_s"])

        dd_vals = row[dd_cols[:n]].to_numpy(dtype=float)
        range_rate = dd_vals * C_LIGHT / delta_t_s

        feats = extract_features(range_rate, cadence_s)
        feats["sat_idx"] = sat_idx
        feature_rows.append(feats)

    features = pd.DataFrame(feature_rows)
    table = features.merge(labels, on="sat_idx", how="left")

    # sat_idx first, then all feat_* columns, then all label_* columns.
    feat_cols = [c for c in table.columns if c.startswith("feat_")]
    label_cols = [c for c in table.columns if c.startswith("label_")]
    table = table[["sat_idx"] + feat_cols + label_cols]

    stats = {
        "n_total":       n_total,
        "n_no_pass":     n_no_pass,
        "n_too_short":   n_too_short,
        "n_non_contig":  n_non_contig,
        "n_final":       len(table),
    }
    return table, stats


if __name__ == "__main__":

    DELAY_PATH  = "output/delay_differences.csv"
    PARAMS_PATH = "output/random_orbital_parameters.csv"
    OUT_PATH    = "output/training_table.csv"

    for p in (DELAY_PATH, PARAMS_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"Required file not found: {p}")

    table, stats = build_training_table(DELAY_PATH, PARAMS_PATH)

    print(f"Loaded {stats['n_total']} satellites from {DELAY_PATH}")
    print(f"  dropped {stats['n_no_pass']:5d} with no pass at all")
    print(f"  dropped {stats['n_too_short']:5d} with fewer than {MIN_SAMPLES} samples")
    print(f"  dropped {stats['n_non_contig']:5d} flagged non-contiguous")
    print(f"  {stats['n_final']} satellites remain in the training table\n")

    feat_cols = [c for c in table.columns if c.startswith("feat_")]
    label_cols = [c for c in table.columns if c.startswith("label_")]
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"Labels   ({len(label_cols)}): {label_cols}\n")

    n_nan_curv = int(table["feat_curvature_mean"].isna().sum())
    print(f"Rows with undefined curvature (n<3 samples): {n_nan_curv}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nTraining table saved to {OUT_PATH}")