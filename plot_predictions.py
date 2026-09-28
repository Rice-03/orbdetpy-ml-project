# =============================================================================
# plot_predictions.py -- predicted-vs-true scatter plots for forest_trainer_v3
# =============================================================================
#
# PURPOSE
# -------
# Reads output/test_predictions_v3.csv (held-out test set, per satellite,
# written by forest_trainer_v3.py) and output/model_results_v3.csv (for the
# R2/RMSE/circ_skill annotations, so nothing is recomputed here) and draws
# one 6-panel figure, one panel per label.
#
# WHY THE CIRCULAR LABELS ARE NOT PLOTTED THE SAME WAY AS THE LINEAR ONES
# ----------------------------------------------------------------------
# sma, ecc, inc are ordinary linear quantities -- a predicted-vs-true
# scatter with a y=x reference line is the direct read.
#
# raan, argp, anom wrap at 360 deg. A predicted-vs-true scatter on the raw
# degree values would show a true value of 359 deg and a prediction of
# 1 deg as a huge miss in the corner of the plot, when it is actually a
# 2-degree error -- exactly the same wraparound problem metrics.py's
# circular_mae/circular_rmse already correct for numerically. The plot used
# here instead is wrapped circular error (pred - true, wrapped to
# (-180, 180] via metrics.circular_error_deg) against the true angle, with
# a y=0 reference line. This shows the same thing a residual plot always
# shows -- random scatter around zero for a genuinely unrecoverable label,
# or a visible pattern if there's real (or spurious) structure -- without
# the wraparound artifact.
#
# LAYOUT
# ------
# 2 rows x 3 columns: top row sma/ecc/inc (pred vs true), bottom row
# raan/argp/anom (wrapped error vs true). Both models overlaid per panel,
# colour-coded, with R2 (linear) or circ_skill (circular) annotated per
# model directly in the panel.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import circular_error_deg

LINEAR_TARGETS = ["label_sma_m", "label_ecc", "label_inc_deg"]
CIRCULAR_TARGETS = ["label_raan_deg", "label_argp_deg", "label_anom_deg"]

# display label, unit conversion factor (applied to both true and pred), axis label
LINEAR_DISPLAY = {
    "label_sma_m":   ("Semi-major axis", 1.0 / 1000.0, "sma (km)"),
    "label_ecc":     ("Eccentricity", 1.0, "ecc"),
    "label_inc_deg": ("Inclination", 1.0, "inc (deg)"),
}
CIRCULAR_DISPLAY = {
    "label_raan_deg": ("RAAN", "raan"),
    "label_argp_deg": ("Argument of perigee", "argp"),
    "label_anom_deg": ("Mean anomaly", "anom"),
}

MODEL_COLORS = {"lightgbm": "#1f77b4", "random_forest": "#ff7f0e"}
MODEL_LABELS = {"lightgbm": "LightGBM", "random_forest": "Random Forest"}


def _annotate(ax, model_name, text, slot):
    """slot 0 = top-left stack, slot 1 = below it -- keeps two models'
    annotations from overlapping."""
    ax.text(
        0.03, 0.97 - slot * 0.09, text, transform=ax.transAxes,
        fontsize=8, va="top", ha="left", color=MODEL_COLORS[model_name],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )


def plot_linear_panel(ax, preds, results, target):
    title, scale, axis_label = LINEAR_DISPLAY[target]
    for slot, model_name in enumerate(MODEL_COLORS):
        sub = preds[(preds["target"] == target) & (preds["model"] == model_name)]
        if sub.empty:
            continue
        y_true = sub["y_true"].to_numpy() * scale
        y_pred = sub["y_pred"].to_numpy() * scale
        ax.scatter(y_true, y_pred, s=14, alpha=0.6,
                   color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])

        r = results[(results["target"] == target) & (results["model"] == model_name)
                     & (results["method"] == "single_split")]
        if not r.empty:
            r2, rmse = r.iloc[0]["r2"], r.iloc[0]["rmse"]
            rmse_disp = rmse * scale
            _annotate(ax, model_name,
                      f"{MODEL_LABELS[model_name]}: R2={r2:.3f}, RMSE={rmse_disp:.3g}", slot)

    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_circular_panel(ax, preds, results, target):
    title, short = CIRCULAR_DISPLAY[target]
    for slot, model_name in enumerate(MODEL_COLORS):
        sub = preds[(preds["target"] == target) & (preds["model"] == model_name)]
        if sub.empty:
            continue
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        err = circular_error_deg(y_true, y_pred)
        ax.scatter(y_true, err, s=14, alpha=0.6,
                   color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])

        r = results[(results["target"] == target) & (results["model"] == model_name)
                     & (results["method"] == "single_split")]
        if not r.empty:
            mae, skill = r.iloc[0]["mae"], r.iloc[0]["circ_skill"]
            _annotate(ax, model_name,
                      f"{MODEL_LABELS[model_name]}: circMAE={mae:.1f}deg, skill={skill:.2f}", slot)

    ax.axhline(0, linestyle="--", color="gray", linewidth=1, zorder=0)
    ax.set_xlim(0, 360)
    ax.set_ylim(-180, 180)
    ax.set_xlabel(f"True {short} (deg)")
    ax.set_ylabel("Wrapped error, pred - true (deg)")
    ax.set_title(f"{title} (negative control)")
    ax.grid(True, alpha=0.3)


def make_figure(predictions_path="output/test_predictions_v3.csv",
                 results_path="output/model_results_v3.csv",
                 out_path="output/predicted_vs_true.png"):
    for p in (predictions_path, results_path):
        if not os.path.exists(p):
            raise SystemExit(f"Required file not found: {p}\nRun forest_trainer_v3.py first.")

    preds = pd.read_csv(predictions_path)
    results = pd.read_csv(results_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Held-out test set: predicted vs true (top) / wrapped error vs true (bottom)")

    for col, target in enumerate(LINEAR_TARGETS):
        plot_linear_panel(axes[0, col], preds, results, target)
    for col, target in enumerate(CIRCULAR_TARGETS):
        plot_circular_panel(axes[1, col], preds, results, target)

    axes[0, 0].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    return out_path


if __name__ == "__main__":
    make_figure()