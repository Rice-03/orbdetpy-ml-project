# =============================================================================
# show_padding_effect.py  –  Demonstrates why trimming padding matters
# =============================================================================
#
# PURPOSE
# -------
# Picks a few real satellites from delay_differences.csv and computes every
# engineered feature TWICE: once correctly (trimmed to num_valid_samples,
# matching what build_dataset.py actually does), and once incorrectly (left
# at full padded width, as if trimming were skipped). Prints both side by
# side so the effect of padding on each feature is visible directly.
#
# This does not change build_dataset.py or its output in any way. It exists
# only to produce a clear, screenshot-able comparison for documentation.
#
# USAGE
# -----
#   python3 show_padding_effect.py                 -> auto-picks 3 examples
#   python3 show_padding_effect.py 107 189 157      -> specific sat_idx values
# =============================================================================

import sys
import numpy as np
import pandas as pd

from build_dataset import extract_features, C_LIGHT


def compare_satellite(row, dd_cols, delta_t_s, cadence_s):
    n = int(row["num_valid_samples"])
    max_len = len(dd_cols)

    dd_full = row[dd_cols].to_numpy(dtype=float)          # full padded row
    rr_full = dd_full * C_LIGHT / delta_t_s                # WITHOUT trimming
    rr_trimmed = rr_full[:n]                               # WITH trimming (correct)

    feats_trimmed = extract_features(rr_trimmed, cadence_s)
    # extract_features assumes evenly-spaced real samples -- calling it on the
    # full padded array is only valid for this side-by-side demonstration,
    # not something build_dataset.py ever actually does.
    feats_untrimmed = extract_features(rr_full, cadence_s)

    return n, max_len, feats_trimmed, feats_untrimmed


def print_comparison(sat_idx, n, max_len, trimmed, untrimmed):
    print(f"\nSatellite {sat_idx}   (real samples: {n} of {max_len} padded columns)")
    print("-" * 55)
    print(f"{'feature':<14} {'WITH trimming':>18} {'WITHOUT trimming':>20}")
    print("-" * 55)
    for key in ("feat_n_samples", "feat_dd_mean"):
        v_trim = trimmed[key]
        v_full = untrimmed[key]
        print(f"{key.replace('feat_', ''):<14} {v_trim:>18.4f} {v_full:>20.4f}")
    print("-" * 55)


if __name__ == "__main__":

    dd = pd.read_csv("output/delay_differences.csv")
    dd_cols = [c for c in dd.columns if c.startswith("dd_")]

    # settings must match generate_params.py / simulationv2.py
    DELTA_T_S = 0.001
    CADENCE_S = 30.0

    if len(sys.argv) > 1:
        sat_indices = [int(x) for x in sys.argv[1:]]
    else:
        # auto-pick 3 illustrative examples: shortest usable, a mid-length
        # pass, and the longest usable, all with num_valid_samples >= 5
        usable = dd[dd.num_valid_samples >= 5].sort_values("num_valid_samples")
        sat_indices = [
            int(usable.iloc[0].sat_idx),                    # shortest
            int(usable.iloc[len(usable) // 2].sat_idx),      # middle length
            int(usable.iloc[-1].sat_idx),                    # longest
        ]
        print(f"No sat_idx given -- auto-picked 3 examples (shortest/middle/longest "
              f"usable pass): {sat_indices}")

    for sat_idx in sat_indices:
        match = dd[dd.sat_idx == sat_idx]
        if match.empty:
            print(f"\n[skip] sat_idx {sat_idx} not found in {('output/delay_differences.csv')}")
            continue
        row = match.iloc[0]
        if row["num_valid_samples"] < 1:
            print(f"\n[skip] sat_idx {sat_idx} has no valid samples (no pass detected)")
            continue

        n, max_len, feats_trimmed, feats_untrimmed = compare_satellite(
            row, dd_cols, DELTA_T_S, CADENCE_S
        )
        print_comparison(sat_idx, n, max_len, feats_trimmed, feats_untrimmed)

    print("\n'WITHOUT trimming' is shown only to illustrate the effect of padding.")
    print("build_dataset.py always trims correctly -- this script does not change")
    print("or reflect its actual behaviour, it only demonstrates why the trimming")
    print("step in extract_features() is necessary.")