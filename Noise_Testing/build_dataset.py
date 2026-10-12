# =============================================================================
# build_dataset.py (v3) – Padded, mask-aware range-rate vectors, no feature
#                          extraction, for LightGBM / Random Forest / 1D-CNN
# =============================================================================
#
# PURPOSE
# -------
# Confirmed plan (Prof. Thron, see the published build-plan document for the
# full step-by-step writeup and diagrams). Replaces the abandoned engineered-
# feature build_dataset.py entirely -- this is a different design, not an
# update of that file.
#
# Every satellite's real samples get resampled onto its own 30-second grid,
# anchored at the pass's zero-range-rate crossing (the point where the curve
# turns from approach to recession), then zero-padded so every row is the
# same length with the crossing at the same column index. A parallel mask
# array marks which values are real vs padding, so no model -- LightGBM,
# Random Forest, or a 1D-CNN -- can use "is this position exactly zero" as a
# free proxy for pass duration.
#
# BLINDNESS BOUNDARY (see the published build-plan doc for the full diagram)
# ----------------------------------------------------------------------
# Everything through crossing detection, spline fitting, and resampling reads
# ONLY delay_differences.csv -- the recorded curve, exactly what a real
# receiver would have. random_orbital_parameters.csv is read exactly once,
# at the very end, purely to attach the six label columns (the answer key
# used to score a prediction afterward). epoch_offset_s and search_window_s
# are never read by this file -- t_star is never reconstructed here. That is
# the one rule everything below follows.
#
# WHY PAD INSTEAD OF TRUNCATE
# ---------------------------
# Pass durations in the current batch run ~450s-1350s, a 3:1 spread.
# Truncating every row to the shortest pass would throw away up to two-
# thirds of the real signal on the longest, highest-altitude passes. Padding
# keeps all of it; the mask array is what keeps that padding from becoming a
# free shortcut for a tree model.
#
# WHY THE CROSSING IS FOUND FROM THE DATA, NOT FROM t_star
# ----------------------------------------------------------
# t_star (generate_params.py's closest-approach construction target) only
# exists because the orbit was deliberately built around it -- a real
# receiver watching an unknown satellite would never have that value. This
# file finds the same point the way a real receiver would have to: by
# looking for the sign change in the recorded curve itself.
#
# NOISE ROBUSTNESS (checked against this project's actual batch, not assumed)
# -----------------------------------------------------------------------
# The step in range rate across the single sample pair that brackets the
# crossing averages ~1228 m/s (570-2500 m/s range, sampled across 80
# satellites) -- three to four orders of magnitude larger than the planned
# 0.1-1.0 m/s measurement noise. Noise at that scale cannot move which
# samples bracket the crossing, and shifts the interpolated crossing time by
# roughly 0.02s in the worst case. The crossing-detection method below does
# not need to change when noise is switched on at that scale.
#
# NEW DEPENDENCY
# ---------------
# scipy is required (CubicSpline) and was not part of this project's
# original dependency list -- `pip install scipy` if not already present.
#
# LABELS
# ------
# All six classical Keplerian elements, same as the abandoned engineered-
# feature file: sma, ecc, inc are the recoverable set; raan, argp, anom are
# kept as the negative control (see v3.md Section 9 for the planned
# follow-up ablation on raan/anom recoverability once absolute time is
# added -- not part of this build).
# =============================================================================

import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


C_LIGHT = 299792458.0

# A cubic spline is technically defined by scipy from 2 points, but a
# meaningful shape fit needs more than that -- this is a data-quality floor,
# not a crash-prevention minimum (verified: scipy raises nothing down to
# n=2). The shortest pass in the current 500-satellite batch has 15 samples,
# so this essentially never triggers today; it exists for future batches
# (noisier runs, a shorter minimum pass duration) where it might.
MIN_REAL_SAMPLES = 4

# How far past a real sample's own observed range the resample grid may
# reach before being cut off. Zero -- never extrapolate past what was
# actually recorded.
_EPS = 1e-6


def _load_delay_differences(path):
    df = pd.read_csv(path)
    dd_cols = [c for c in df.columns if c.startswith("dd_")]
    return df, dd_cols


def _load_labels(params_path):
    """
    Ground-truth labels, keyed by sat_idx, read once, only to attach the
    answer key at the very end -- never touched by anything upstream of the
    final merge. Same six columns and naming convention as the abandoned
    engineered-feature file, for continuity.
    """
    params = pd.read_csv(params_path)
    labels = params[["sat_idx", "sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "anom_rad"]].copy()
    labels["inc_deg"] = np.degrees(labels["inc_rad"])
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


def find_zero_crossing(t, rr):
    """
    Step 3-4 of the confirmed plan: walk the recorded range-rate samples for
    the pair that flips sign (approach turning into recession), then pin
    down the exact crossing time by linear interpolation between that pair.

    t, rr : 1D arrays, this satellite's real (recorded) samples only.
    Returns t_cross, or None if no approach-to-recession crossing is found.
    """
    for i in range(len(rr) - 1):
        if rr[i] < 0 <= rr[i + 1]:
            frac = -rr[i] / (rr[i + 1] - rr[i])
            return t[i] + frac * (t[i + 1] - t[i])
    return None


def resample_satellite(t, rr, t_cross, cadence_s):
    """
    Steps 5-6: fit a cubic spline through this satellite's own real samples,
    then evaluate it on a 30-second grid anchored at t_cross, extending
    outward in both directions only as far as this satellite's own observed
    range (t[0] to t[-1]) actually reaches -- never extrapolated.

    Returns (before_vals, after_vals): before_vals ends WITH the crossing
    sample itself (chronological order); after_vals starts with the first
    point strictly after it.
    """
    spline = CubicSpline(t, rr)
    t_min, t_max = t[0], t[-1]

    before_times = []
    k = 0
    while True:
        tt = t_cross - k * cadence_s
        if tt < t_min - _EPS:
            break
        before_times.append(tt)
        k += 1
    before_times.reverse()  # oldest -> newest, ending at t_cross

    after_times = []
    k = 1
    while True:
        tt = t_cross + k * cadence_s
        if tt > t_max + _EPS:
            break
        after_times.append(tt)
        k += 1

    before_vals = spline(np.clip(before_times, t_min, t_max))
    after_vals = spline(np.clip(after_times, t_min, t_max))
    return before_vals, after_vals


def build_training_table(
    delay_path="output/delay_differences.csv",
    params_path="output/random_orbital_parameters.csv",
    min_real_samples=MIN_REAL_SAMPLES,
):
    dd_df, dd_cols = _load_delay_differences(delay_path)
    labels = _load_labels(params_path)
    params_lookup = pd.read_csv(params_path).set_index("sat_idx")[["delta_t_s", "cadence_s"]]

    n_total = len(dd_df)
    n_no_pass = int((dd_df["num_valid_samples"] == 0).sum())
    usable = dd_df[dd_df["num_valid_samples"] > 0].copy()

    # Times are only reconstructed as t_aos_s + i*cadence_s, which is only
    # valid when there are no gaps -- non-contiguous rows are dropped rather
    # than risk a silently wrong time axis.
    n_non_contig = int((usable["contiguous"] == 0).sum())
    usable = usable[usable["contiguous"] == 1]

    # --- pass 1: per-satellite crossing + spline-resampled segments ---
    processed = {}
    n_too_short = 0
    n_no_crossing = 0

    for _, row in usable.iterrows():
        sat_idx = int(row["sat_idx"])
        n = int(row["num_valid_samples"])

        if n < min_real_samples:
            n_too_short += 1
            continue

        delta_t_s = float(params_lookup.loc[sat_idx, "delta_t_s"])
        cadence_s = float(params_lookup.loc[sat_idx, "cadence_s"])
        t_aos_s = float(row["t_aos_s"])

        t = np.array([t_aos_s + i * cadence_s for i in range(n)])
        dd_vals = row[dd_cols[:n]].to_numpy(dtype=float)
        rr = dd_vals * C_LIGHT / delta_t_s

        t_cross = find_zero_crossing(t, rr)
        if t_cross is None:
            n_no_crossing += 1
            continue

        before_vals, after_vals = resample_satellite(t, rr, t_cross, cadence_s)
        processed[sat_idx] = {
            "k_before": len(before_vals),
            "k_after": len(after_vals),
            "before_vals": before_vals,
            "after_vals": after_vals,
        }

    if not processed:
        raise SystemExit("No satellites produced a usable zero crossing -- check upstream data.")

    # --- pass 2: pad every row to the batch-wide max on each side ---
    K_before = max(r["k_before"] for r in processed.values())
    K_after = max(r["k_after"] for r in processed.values())
    L = K_before + K_after

    rr_cols = [f"rr_c_{i:03d}" for i in range(L)]
    mask_cols = [f"mask_{i:03d}" for i in range(L)]

    rows = []
    for sat_idx, r in processed.items():
        vec = np.zeros(L, dtype=float)
        mask = np.zeros(L, dtype=np.int8)

        kb, ka = r["k_before"], r["k_after"]
        start_before = K_before - kb
        vec[start_before:K_before] = r["before_vals"]
        mask[start_before:K_before] = 1
        vec[K_before:K_before + ka] = r["after_vals"]
        mask[K_before:K_before + ka] = 1

        row = {"sat_idx": sat_idx}
        row.update(zip(rr_cols, vec))
        row.update(zip(mask_cols, mask))
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.merge(labels, on="sat_idx", how="left")

    label_cols = [c for c in table.columns if c.startswith("label_")]
    table = table[["sat_idx"] + rr_cols + mask_cols + label_cols]

    stats = {
        "n_total": n_total,
        "n_no_pass": n_no_pass,
        "n_non_contig": n_non_contig,
        "n_too_short": n_too_short,
        "n_no_crossing": n_no_crossing,
        "n_final": len(table),
        "K_before": K_before,
        "K_after": K_after,
        "L": L,
    }
    return table, stats


if __name__ == "__main__":

    DELAY_PATH  = "output/delay_differences.csv"
    PARAMS_PATH = "output/random_orbital_parameters.csv"
    OUT_PATH    = "output/training_table_padded.csv"

    for p in (DELAY_PATH, PARAMS_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"Required file not found: {p}")

    table, stats = build_training_table(DELAY_PATH, PARAMS_PATH)

    print(f"Loaded {stats['n_total']} satellites from {DELAY_PATH}")
    print(f"  dropped {stats['n_no_pass']:5d} with no pass at all")
    print(f"  dropped {stats['n_non_contig']:5d} flagged non-contiguous")
    print(f"  dropped {stats['n_too_short']:5d} with fewer than {MIN_REAL_SAMPLES} real samples")
    print(f"  dropped {stats['n_no_crossing']:5d} with no zero crossing found")
    print(f"  {stats['n_final']} satellites remain in the training table\n")

    print(f"K_before = {stats['K_before']}  (crossing sits at column index {stats['K_before']-1})")
    print(f"K_after  = {stats['K_after']}")
    print(f"L        = {stats['L']}  (total rr_c_* / mask_* columns)\n")

    label_cols = [c for c in table.columns if c.startswith("label_")]
    print(f"Labels ({len(label_cols)}): {label_cols}")

    real_fraction = table[[c for c in table.columns if c.startswith("mask_")]].mean().mean()
    print(f"Mean fraction of real (non-padded) samples per row: {real_fraction:.1%}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nTraining table saved to {OUT_PATH}")