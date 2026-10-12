# =============================================================================
# forest_trainer_v3.py -- LightGBM + Random Forest on the padded, mask-aware
#                          range-rate vectors from build_dataset_v3.py
# =============================================================================
#
# PURPOSE
# -------
# Reads output/training_table_padded.csv and trains a separate LightGBM
# regressor and a separate Random Forest regressor for each of the six
# labels, with no hyperparameter tuning -- same ceiling/floor check as the
# original forest_trainer.py, on the new padded-vector features instead of
# the abandoned engineered-feature table.
#
# WHAT CHANGED FROM forest_trainer.py
# --------------------------------------
# 1. Input: rr_c_* + mask_* columns from training_table_padded.csv, instead
#    of feat_* columns from training_table.csv.
# 2. Angular labels (raan, argp, anom): trained as sin/cos pairs and
#    recombined via atan2 (CircularAngleRegressor below), instead of raw
#    degrees. Scored with the wrapped circular metrics in metrics.py instead
#    of raw R2/RMSE, per the project's own noted principle that those aren't
#    reliable for a wraparound quantity.
# 3. Split: one fixed 80/20 split, keyed by sat_idx, saved to
#    output/train_test_split.csv (splits.py) so this run and any later CNN
#    run score against the exact same held-out satellites. The old file's
#    train_test_split() call drew a fresh random split every run.
# 4. CV: 5-fold CV now runs strictly on the TRAIN partition (the fixed 80%),
#    never touching the held-out test set. The old file's CV ran over the
#    whole table -- correct at the time, but not what the project's stated
#    evaluation method (CV inside the training partition) calls for once a
#    real held-out set exists.
#
# WHY BOTH MODELS, WHY ONE MODEL PER TARGET, WHY NOT NAMED lightgbm.py
# ------------------------------------------------------------------------
# Unchanged reasoning from forest_trainer.py -- see that file's header if
# needed. Not repeated here to keep this file focused on what's different.
#
# WHAT THIS PHASE DELIBERATELY DOES NOT DO
# -------------------------------------------
# No hyperparameter search, no feature importances, no plots. Same scope
# limit as the original file.
# =============================================================================

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, clone

from splits import make_or_load_split, RANDOM_STATE
from metrics import linear_metrics, circular_metrics


LINEAR_TARGETS = ["label_sma_m", "label_ecc", "label_inc_deg"]
CIRCULAR_TARGETS = ["label_raan_deg", "label_argp_deg", "label_anom_deg"]
ALL_TARGETS = LINEAR_TARGETS + CIRCULAR_TARGETS

MODELS = {
    "lightgbm": lambda seed: LGBMRegressor(random_state=seed, verbosity=-1),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=300, random_state=seed),
}

CV_FOLDS = 5
N_SHUFFLES = 20  # repeats for circular_skill's shuffled-baseline estimate


class CircularAngleRegressor(BaseEstimator, RegressorMixin):
    """
    Two copies of base_estimator -- one fit on sin(angle), one on
    cos(angle) (angle in degrees in, radians internally) -- recombined via
    atan2 at predict time, wrapped back to [0, 360) degrees.

    Subclasses BaseEstimator/RegressorMixin (get_params introspects
    __init__, base_estimator is cloned via sklearn's clone()) so it has a
    normal fit/predict interface -- build_estimator() and score() use it
    exactly like a plain regressor, no separate codepath for the circular
    labels in held_out_test() or cross_validate_in_train().
    """
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y_rad = np.radians(np.asarray(y, dtype=float))
        self.model_sin_ = clone(self.base_estimator)
        self.model_cos_ = clone(self.base_estimator)
        self.model_sin_.fit(X, np.sin(y_rad))
        self.model_cos_.fit(X, np.cos(y_rad))
        return self

    def predict(self, X):
        s = self.model_sin_.predict(X)
        c = self.model_cos_.predict(X)
        return np.degrees(np.arctan2(s, c)) % 360.0


def is_circular(target):
    return target in CIRCULAR_TARGETS


def load_table(path="output/training_table_padded.csv"):
    if not os.path.exists(path):
        raise SystemExit(
            f"Training table not found: {path}\nRun build_dataset_v3.py first."
        )
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("rr_c_") or c.startswith("mask_")]
    return df, feat_cols


def build_estimator(model_name, target, seed):
    base = MODELS[model_name](seed)
    return CircularAngleRegressor(base_estimator=base) if is_circular(target) else base


def score(y_true, y_pred, target):
    """Returns a dict with r2, rmse, mae, circ_skill -- the two metric
    families share these four keys so downstream code doesn't branch on
    target type; whichever family doesn't apply is left as NaN."""
    if is_circular(target):
        m = circular_metrics(y_true, y_pred, n_shuffles=N_SHUFFLES, random_state=RANDOM_STATE)
        m["r2"] = float("nan")  # not meaningful on wrapped angles -- see metrics.py
        return m
    m = linear_metrics(y_true, y_pred)
    m["circ_skill"] = float("nan")
    return m


def held_out_test(df, feat_cols, target, model_name, train_ids, test_ids, seed=RANDOM_STATE):
    """Returns (metrics_dict, predictions_df). predictions_df is per-satellite
    (sat_idx, target, model, y_true, y_pred) on the held-out test set only --
    this is what plot_predictions.py reads to draw the scatter plots."""
    sub = df.dropna(subset=[target])
    n_dropped = len(df) - len(sub)

    train_mask = sub["sat_idx"].isin(train_ids)
    test_mask = sub["sat_idx"].isin(test_ids)
    X_train, y_train = sub.loc[train_mask, feat_cols], sub.loc[train_mask, target]
    X_test, y_test = sub.loc[test_mask, feat_cols], sub.loc[test_mask, target]
    test_sat_idx = sub.loc[test_mask, "sat_idx"].to_numpy()

    model = build_estimator(model_name, target, seed)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    m = score(y_test.to_numpy(), pred_test, target)
    m.update({
        "target": target, "model_name": model_name,
        "n_train": len(X_train), "n_test": len(X_test), "n_dropped_nan": n_dropped,
    })
    predictions = pd.DataFrame({
        "sat_idx": test_sat_idx,
        "target": target,
        "model": model_name,
        "y_true": y_test.to_numpy(),
        "y_pred": pred_test,
    })
    return m, predictions


def _agg_stat(values):
    """Mean and std across folds, NaN-safe: if every fold's value is NaN
    (the metric family that doesn't apply to this target -- r2 for a
    circular target, circ_skill for a linear one), returns (NaN, NaN)
    instead of letting np.nanmean warn on an all-NaN slice."""
    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))


def cross_validate_in_train(df, feat_cols, target, model_name, train_ids,
                             n_folds=CV_FOLDS, seed=RANDOM_STATE):
    """5-fold CV strictly on the TRAIN partition, scored PER FOLD and then
    aggregated as mean +/- std across the 5 folds -- not pooled into one
    number via cross_val_predict. Pooling can hide a fold that scores very
    differently from the rest (see v3.md Section 16 for why this changed);
    scoring per fold and keeping the spread is what surfaces that. The
    held-out test set from held_out_test() is never referenced here."""
    train_only = df[df["sat_idx"].isin(train_ids)]
    sub = train_only.dropna(subset=[target])
    n_dropped = len(train_only) - len(sub)

    X = sub[feat_cols].reset_index(drop=True)
    y = sub[target].reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = build_estimator(model_name, target, seed)
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)

        fold_scores.append(score(y_val.to_numpy(), pred_val, target))

    m = {}
    for key in ("r2", "rmse", "mae", "circ_skill"):
        mean_v, std_v = _agg_stat([fs[key] for fs in fold_scores])
        m[key] = mean_v
        m[f"{key}_std"] = std_v

    m.update({
        "target": target, "model_name": model_name,
        "n": len(sub), "n_dropped_nan": n_dropped,
    })
    return m


if __name__ == "__main__":

    df, feat_cols = load_table()
    n_rr = sum(c.startswith("rr_c_") for c in feat_cols)
    n_mask = sum(c.startswith("mask_") for c in feat_cols)
    print(f"Loaded {len(df)} rows, {len(feat_cols)} features "
          f"({n_rr} rr_c_* + {n_mask} mask_*) from output/training_table_padded.csv\n")

    split_df = make_or_load_split(df["sat_idx"])
    train_ids = set(split_df.loc[split_df["split"] == "train", "sat_idx"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "sat_idx"])
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test satellites "
          f"(output/train_test_split.csv)\n")

    # --- held-out test set, once per target x model ---
    test_results = []
    test_predictions = []
    for target in ALL_TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target} not in table")
            continue
        for model_name in MODELS:
            m, preds = held_out_test(df, feat_cols, target, model_name, train_ids, test_ids)
            test_results.append(m)
            test_predictions.append(preds)

    predictions_path = "output/test_predictions_v3.csv"
    pd.concat(test_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    print(f"Saved per-satellite held-out test predictions to {predictions_path} "
          f"(input for plot_predictions.py)\n")

    print(f"{'target':>28} {'model':>14} {'n_train':>8} {'n_test':>7} "
          f"{'R2':>8} {'RMSE':>12} {'MAE':>12} {'circ_skill':>11}")
    print("-" * 112)
    for target in ALL_TARGETS:
        rows = [r for r in test_results if r["target"] == target]
        for r in rows:
            r2_str = f"{r['r2']:.3f}" if not np.isnan(r["r2"]) else "  --  "
            skill_str = f"{r['circ_skill']:.3f}" if not np.isnan(r["circ_skill"]) else "  --  "
            print(f"{r['target']:>28} {r['model_name']:>14} {r['n_train']:>8} {r['n_test']:>7} "
                  f"{r2_str:>8} {r['rmse']:>12.4f} {r['mae']:>12.4f} {skill_str:>11}")
        if rows and rows[0]["n_dropped_nan"]:
            print(f"    [note] {rows[0]['n_dropped_nan']} rows dropped (NaN in this target)")
        print()

    print("Linear labels (sma, ecc, inc): R2 near 0 or negative means no better than the mean.")
    print("Circular labels (raan, argp, anom): R2 is left blank (not meaningful on wrapped")
    print("angles); circ_skill near 0 means no better than guessing a random label from the")
    print("batch, negative means worse than that.\n")

    print("=" * 112)
    print("5-FOLD CROSS-VALIDATION ON THE TRAIN PARTITION ONLY "
          "(the held-out test set above is never touched here)")
    print("=" * 112)

    cv_results = []
    for target in ALL_TARGETS:
        if target not in df.columns:
            continue
        for model_name in MODELS:
            cv_results.append(
                cross_validate_in_train(df, feat_cols, target, model_name, train_ids)
            )

    print(f"\n{'target':>28} {'model':>14} {'n':>7} "
          f"{'R2 (+/-std)':>16} {'RMSE (+/-std)':>18} {'MAE (+/-std)':>18} "
          f"{'circ_skill (+/-std)':>20}")
    print("-" * 132)
    for target in ALL_TARGETS:
        rows = [r for r in cv_results if r["target"] == target]
        for r in rows:
            if not np.isnan(r["r2"]):
                r2_str = f"{r['r2']:.3f}+/-{r['r2_std']:.3f}"
            else:
                r2_str = "  --  "
            if not np.isnan(r["circ_skill"]):
                skill_str = f"{r['circ_skill']:.3f}+/-{r['circ_skill_std']:.3f}"
            else:
                skill_str = "  --  "
            rmse_str = f"{r['rmse']:.4f}+/-{r['rmse_std']:.4f}"
            mae_str = f"{r['mae']:.4f}+/-{r['mae_std']:.4f}"
            print(f"{r['target']:>28} {r['model_name']:>14} {r['n']:>7} "
                  f"{r2_str:>16} {rmse_str:>18} {mae_str:>18} {skill_str:>20}")
        if rows and rows[0]["n_dropped_nan"]:
            print(f"    [note] {rows[0]['n_dropped_nan']} rows dropped (NaN in this target)")
        print()

    print("Mean +/- std across the 5 folds (scored per fold, not pooled) -- a large std")
    print("relative to the mean means fold-to-fold performance is inconsistent, which a single")
    print("pooled CV number would hide. The held-out test numbers above are still the ones to")
    print("cite as generalization performance; these are for checking that stability.")

    # =========================================================================
    # Save results -- "method" values (single_split / cv) kept identical to
    # the old forest_trainer.py for leo_tek_dashboard.py compatibility.
    # NOTE: "single_split" here is the fixed, saved 80/20 split
    # (output/train_test_split.csv), not a fresh random split each run, as
    # it was in the old file -- flagged in case the dashboard's own text
    # assumes the old meaning.
    #
    # Columns added vs. the old file: circ_skill, and (cv rows only)
    # r2_std/rmse_std/mae_std/circ_skill_std -- the fold-to-fold spread
    # behind the cv row's mean (see cross_validate_in_train's docstring).
    # NaN for single_split rows, which have no folds to spread across.
    #
    # Saved as model_results_v3.csv, not model_results.csv, so this run
    # never silently overwrites results produced by the abandoned
    # engineered-feature pipeline -- same reasoning as
    # training_table_padded.csv vs. training_table.csv.
    # =========================================================================
    os.makedirs("output", exist_ok=True)

    out_rows = []
    for r in test_results:
        out_rows.append({
            "method": "single_split", "model": r["model_name"], "target": r["target"],
            "n": r["n_test"], "n_train": r["n_train"],
            "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"], "circ_skill": r["circ_skill"],
            "r2_std": np.nan, "rmse_std": np.nan, "mae_std": np.nan, "circ_skill_std": np.nan,
        })
    for r in cv_results:
        out_rows.append({
            "method": "cv", "model": r["model_name"], "target": r["target"],
            "n": r["n"], "n_train": np.nan,
            "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"], "circ_skill": r["circ_skill"],
            "r2_std": r["r2_std"], "rmse_std": r["rmse_std"],
            "mae_std": r["mae_std"], "circ_skill_std": r["circ_skill_std"],
        })

    results_path = "output/model_results_v3.csv"
    pd.DataFrame(out_rows).to_csv(results_path, index=False)
    print(f"\nSaved model results to {results_path}")