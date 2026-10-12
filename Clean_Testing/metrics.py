# =============================================================================
# metrics.py -- evaluation metrics for linear and circular (angular) labels
# =============================================================================
#
# WHY THIS EXISTS
# -----------------
# sma, ecc, and inc_deg are ordinary linear quantities -- R2/RMSE/MAE apply
# directly. raan_deg, argp_deg, and anom_deg are angles that wrap at 360 deg:
# a prediction of 359 deg against a true value of 1 deg is a 2-degree error,
# not a 358-degree one. Per the project's own noted principle, RMSE and R2
# on the raw degree values are not reliable for these three labels -- this
# module is the wrapped replacement.
#
# CIRCULAR SKILL
# ----------------
# A single circular MAE number does not say whether the model beats having
# no information at all. circular_skill compares the model's circular MAE
# against the circular MAE of the true labels against a permutation of
# themselves (i.e. "guess a randomly-drawn label from the batch for every
# row"), averaged over several permutations for stability at this sample
# size. skill = 1 - model_MAE / baseline_MAE: 0 means no better than
# guessing from the label distribution, 1 means zero error, negative means
# worse than guessing.
# =============================================================================

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def linear_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def circular_error_deg(y_true_deg, y_pred_deg):
    """Wrapped signed difference (pred - true), in (-180, 180] degrees."""
    y_true_deg = np.asarray(y_true_deg, dtype=float)
    y_pred_deg = np.asarray(y_pred_deg, dtype=float)
    return (y_pred_deg - y_true_deg + 180.0) % 360.0 - 180.0


def circular_mae(y_true_deg, y_pred_deg):
    return float(np.mean(np.abs(circular_error_deg(y_true_deg, y_pred_deg))))


def circular_rmse(y_true_deg, y_pred_deg):
    err = circular_error_deg(y_true_deg, y_pred_deg)
    return float(np.sqrt(np.mean(err ** 2)))


def circular_skill(y_true_deg, y_pred_deg, n_shuffles=20, random_state=42):
    """
    1 - (model circular MAE) / (circular MAE of y_true against a shuffled
    copy of itself, averaged over n_shuffles permutations). See module
    docstring. Returns NaN if the baseline MAE is exactly zero (degenerate
    label set -- should not occur here).
    """
    y_true_deg = np.asarray(y_true_deg, dtype=float)
    model_mae = circular_mae(y_true_deg, y_pred_deg)

    rng = np.random.RandomState(random_state)
    baseline_maes = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(y_true_deg)
        baseline_maes.append(circular_mae(y_true_deg, shuffled))
    baseline_mae = float(np.mean(baseline_maes))

    if baseline_mae == 0.0:
        return float("nan")
    return 1.0 - model_mae / baseline_mae


def circular_metrics(y_true_deg, y_pred_deg, n_shuffles=20, random_state=42):
    return {
        "rmse": circular_rmse(y_true_deg, y_pred_deg),
        "mae": circular_mae(y_true_deg, y_pred_deg),
        "circ_skill": circular_skill(y_true_deg, y_pred_deg, n_shuffles, random_state),
    }