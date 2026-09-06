# =============================================================================
# forest_trainer.py  --  Phase 1+2: LightGBM + Random Forest baselines
# =============================================================================
#
# PURPOSE
# -------
# Reads output/training_table.csv (produced by build_dataset.py) and trains a
# separate LightGBM regressor AND a separate Random Forest regressor for each
# label column, with NO hyperparameter tuning. This is the ceiling/floor check
# before anything is optimised: whether the engineered features carry real
# signal for each target at all, and whether the two model families agree.
#
# WHY BOTH MODELS
# -----------------
# If LightGBM and Random Forest broadly agree on which targets are learnable,
# that's evidence the signal (or lack of it) is a property of the DATA, not an
# artifact of one model's quirks. If they disagree sharply, that's worth
# investigating before trusting either number.
#
# WHY ONE MODEL PER TARGET, NOT A SINGLE MULTI-OUTPUT MODEL
# ------------------------------------------------------------
# LightGBM has no native multi-output regression. Random Forest does
# (RandomForestRegressor accepts a 2D target directly), but it's trained
# per-target here too, to keep the comparison symmetric and per-target
# diagnostics (R2, RMSE, later: feature importances) directly accessible.
#
# WHY NOT NAMED lightgbm.py
# ---------------------------
# Python puts a script's own directory first on the import path. A file named
# lightgbm.py would shadow the real `lightgbm` package on `import lightgbm`,
# causing a self-import / crash. This file avoids the collision (and now that
# it trains two model families rather than one, forest_trainer.py describes
# its scope better than a lightgbm-specific name would).
#
# SPLIT
# -----
# Plain 80/20 train/test, fixed random_state for reproducibility. No reserved
# third partition. An earlier version of this script carved out an additional
# 15% as a "validation set", but on checking, that isn't what a single-split
# method conventionally means (train/test and train/validation/test are
# named as two separate methodologies, not one with an optional extra piece),
# and scikit-learn's own documented workflow doesn't use a static reserved
# split for tuning either -- it cross-validates the training set directly.
# Since no tuning was actually being done against that reserved chunk, it was
# sitting unused for no real reason. Removed; if/when tuning is added later,
# it will cross-validate the training set here, matching that workflow,
# rather than reintroducing a separate static split.
#
# WHAT THIS PHASE DELIBERATELY DOES NOT DO
# -------------------------------------------
# No hyperparameter search, no feature importances, no plots. Those are later
# phases, once this baseline confirms the pipeline and features are sound.
# =============================================================================

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error


TARGETS = ["label_sma_m", "label_ecc", "label_inc_deg",
           "label_raan_deg", "label_argp_deg", "label_anom_deg"]

MODELS = {
    "lightgbm": lambda seed: LGBMRegressor(random_state=seed, verbosity=-1),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=300, random_state=seed),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_table(path="output/training_table.csv"):
    if not os.path.exists(path):
        raise SystemExit(
            f"Training table not found: {path}\nRun build_dataset.py first."
        )
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    return df, feat_cols


def split_data(df, feat_cols, target, random_state=RANDOM_STATE):
    """
    Plain 80/20 train/test split. Rows with a NaN in this specific target are
    dropped before splitting (relevant for a label if its source file wasn't
    available when build_dataset.py ran).
    """
    sub = df.dropna(subset=[target])
    X = sub[feat_cols]
    y = sub[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    n_dropped = len(df) - len(sub)
    return X_train, X_test, y_train, y_test, n_dropped


def train_baseline(df, feat_cols, target, model_name, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test, n_dropped = split_data(
        df, feat_cols, target, random_state
    )

    model = MODELS[model_name](random_state)
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    rmse = float(np.sqrt(((y_test - pred_test) ** 2).mean()))
    mae = mean_absolute_error(y_test, pred_test)

    return {
        "target": target,
        "model_name": model_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_dropped_nan": n_dropped,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "model": model,
    }


def cross_validate_target(df, feat_cols, target, model_name, n_folds=CV_FOLDS, random_state=RANDOM_STATE):
    """
    5-fold CV, added alongside (not instead of) the single 80/20 split.
    Every row gets used as a test point exactly once across the 5 folds, so
    the result doesn't depend on which rows happened to land in one held-out
    slice -- this is what caught RAAN scoring +0.19 on one lucky single split
    while consistently scoring negative under CV. Kept side by side with the
    single-split table on purpose, so that gap stays visible rather than
    silently disappearing once CV replaces it.
    """
    sub = df.dropna(subset=[target])
    X = sub[feat_cols].to_numpy()
    y = sub[target].to_numpy()
    n_dropped = len(df) - len(sub)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model = MODELS[model_name](random_state)
    pred = cross_val_predict(model, X, y, cv=kf)

    r2 = r2_score(y, pred)
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    mae = float(np.abs(y - pred).mean())

    return {
        "target": target,
        "model_name": model_name,
        "n": len(sub),
        "n_dropped_nan": n_dropped,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


if __name__ == "__main__":

    df, feat_cols = load_table()
    print(f"Loaded {len(df)} rows, {len(feat_cols)} features from output/training_table.csv\n")

    results = []
    for target in TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target} not in table")
            continue
        for model_name in MODELS:
            res = train_baseline(df, feat_cols, target, model_name)
            results.append(res)

    print(f"{'target':>28} {'model':>14} {'n_train':>8} {'n_test':>7} {'R2':>8} {'RMSE':>14} {'MAE':>14}")
    print("-" * 100)
    for target in TARGETS:
        target_results = [r for r in results if r["target"] == target]
        for r in target_results:
            print(f"{r['target']:>28} {r['model_name']:>14} {r['n_train']:>8} {r['n_test']:>7} "
                  f"{r['r2']:>8.3f} {r['rmse']:>14.4f} {r['mae']:>14.4f}")
        if target_results and target_results[0]["n_dropped_nan"]:
            print(f"    [note] {target_results[0]['n_dropped_nan']} rows dropped (NaN in this target)")
        print()

    print("R2 near 0 or negative means the model is doing no better than predicting")
    print("the mean -- not a bug, a genuine identifiability signal for that target.")
    print("If LightGBM and Random Forest disagree sharply on a target, investigate")
    print("before trusting either number.\n")

    print("=" * 100)
    print(f"5-FOLD CROSS-VALIDATION  (every row used as a test point once, "
          f"more reliable than a single split at this sample size)")
    print("=" * 100)

    cv_results = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        for model_name in MODELS:
            cv_results.append(cross_validate_target(df, feat_cols, target, model_name))

    print(f"\n{'target':>28} {'model':>14} {'n':>7} {'R2':>8} {'RMSE':>14} {'MAE':>14}")
    print("-" * 92)
    for target in TARGETS:
        target_cv = [r for r in cv_results if r["target"] == target]
        for r in target_cv:
            print(f"{r['target']:>28} {r['model_name']:>14} {r['n']:>7} "
                  f"{r['r2']:>8.3f} {r['rmse']:>14.4f} {r['mae']:>14.4f}")
        if target_cv and target_cv[0]["n_dropped_nan"]:
            print(f"    [note] {target_cv[0]['n_dropped_nan']} rows dropped (NaN in this target)")
        print()

    print("These are the numbers to cite -- averaged over 5 independent test folds")
    print("rather than one split, so a single lucky/unlucky slice of rows can't")
    print("swing the result the way it can above.")

    # =========================================================================
    # Save results to CSV -- this is what leo_tek_dashboard.py reads to render
    # the "Model results" page. Both the single-split and the CV numbers are
    # written to the same file, distinguished by the "method" column, so the
    # dashboard can let the user flip between them.
    #
    # The "model" object itself (an actual fitted LGBMRegressor / RandomForest-
    # Regressor instance) is dropped here -- it isn't CSV-serialisable and
    # isn't needed by the dashboard, which only reads the scored metrics.
    # =========================================================================
    os.makedirs("output", exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "method": "single_split",
            "model": r["model_name"],
            "target": r["target"],
            "n": r["n_test"],
            "n_train": r["n_train"],
            "r2": r["r2"],
            "rmse": r["rmse"],
            "mae": r["mae"],
        })
    for r in cv_results:
        rows.append({
            "method": "cv",
            "model": r["model_name"],
            "target": r["target"],
            "n": r["n"],
            "n_train": np.nan,
            "r2": r["r2"],
            "rmse": r["rmse"],
            "mae": r["mae"],
        })

    results_path = "output/model_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nSaved model results to {results_path}")