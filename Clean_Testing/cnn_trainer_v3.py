# =============================================================================
# cnn_trainer_v3.py -- 1D convolutional neural network on the padded,
#                      mask-aware range-rate vectors (build_dataset_v3.py)
# =============================================================================
#
# PURPOSE
# -------
# The learned-feature counterpart to forest_trainer_v3.py. Same input table
# (output/training_table_padded.csv), same fixed 80/20 split
# (output/train_test_split.csv via splits.py), same 5 CV folds inside the
# training partition, same scoring code (score() imported from
# forest_trainer_v3.py), same one-model-per-label design. The only thing
# that changes between the trees and this file is the model, so any
# difference in the results is attributable to the model.
#
# INPUT TO THE NETWORK
# ----------------------
# Each satellite becomes a (2, L) array, L = 44 on the current batch:
#   channel 0: rr_c_* / RR_SCALE   (range rate, divided by one fixed
#                                   constant for the whole dataset)
#   channel 1: mask_*              (1 = real sample, 0 = padding)
# RR_SCALE is fixed (8000 m/s, the LEO range-rate band simv3.py already
# checks against), NOT a per-row normalisation: the curve's amplitude
# carries altitude and pass-geometry information, and per-row scaling
# would erase it.
#
# ARCHITECTURE (DopplerCNN)
# ---------------------------
#   Conv1d(2 -> 16, k=5, pad=2) + ReLU       length 44
#   Conv1d(16 -> 32, k=5, pad=2) + ReLU      length 44
#   MaxPool1d(2)                             length 22
#   Conv1d(32 -> 32, k=3, pad=1) + ReLU      length 22
#   Flatten                                  32 x 22 = 704 values
#   Dropout(0.2) -> Linear(704 -> 64) + ReLU -> Linear(64 -> n_out)
# n_out = 1 for sma/ecc/inc (standardised target), 2 for raan/argp/anom
# (sin and cos of the angle, recombined with atan2).
# ~51k trainable parameters. Deliberately NO global average pooling: the
# crossing sits at a fixed column, so column position means time relative
# to closest approach, and global pooling would discard it. Flatten keeps
# it.
#
# TRAINING (two-stage early stopping, all inside the training data)
# -------------------------------------------------------------------
# Stage 1: carve 20% of the model's own training data out as an
#          early-stopping subset. Train on the other 80% (MSE loss, AdamW),
#          evaluating the early-stopping subset after every epoch; stop
#          after PATIENCE epochs with no improvement and record the best
#          epoch count. PATIENCE = 150 and MAX_EPOCHS = 2000, not smaller:
#          a first run with PATIENCE = 40 / MAX_EPOCHS = 400 let two of
#          the five inclination CV folds stop at epochs 8 and 27 on an early
#          plateau and score R2 near zero, while the folds that trained past
#          360 epochs scored 0.82-0.89 (the multiple-local-minima behaviour
#          of validation curves described in [10]). Chosen from the CV folds
#          inside the training partition only -- the held-out test set
#          played no part (see v3.md Section 17).
# Stage 2: re-initialise and retrain on ALL of the training data for that
#          many epochs. The final model therefore learns from exactly the
#          same satellites the trees learned from (400 for the held-out
#          test, the fold's training part inside CV), not 80% of them.
# The held-out test satellites are never used for any training decision.
# MSE is the training objective only; it is not a reported metric. Its
# square root is RMSE, which is reported, so the objective and the
# headline error metric measure the same thing.
#
# REFERENCES (full explanation in v3.md Section 17)
# ---------------------------------------------------
# [1]  LeCun, Bottou, Bengio, Haffner (1998). Gradient-based learning applied
#      to document recognition. Proc. IEEE 86(11), 2278-2324.  (CNNs)
# [2]  Kiranyaz, Avci, Abdeljaber, Ince, Gabbouj, Inman (2021). 1D
#      convolutional neural networks and applications: A survey. Mechanical
#      Systems and Signal Processing 151, 107398.  (1D CNNs on signals)
# [3]  Goodfellow, Bengio, Courville (2016). Deep Learning. MIT Press.
#      Ch. 9 (convolution, parameter sharing, pooling), Sec. 7.8 and
#      Algorithm 7.2 (early stopping, retrain on all training data),
#      Sec. 6.2 (MSE loss for regression).
# [4]  Dumoulin, Visin (2016). A guide to convolution arithmetic for deep
#      learning. arXiv:1603.07285.  (kernel size, padding, stride, output
#      length, receptive field)
# [5]  Nair, Hinton (2010). Rectified linear units improve restricted
#      Boltzmann machines. ICML.  (ReLU)
# [6]  Lin, Chen, Yan (2014). Network in network. ICLR.  (global average
#      pooling, cited for why it is NOT used here)
# [7]  Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014).
#      Dropout: A simple way to prevent neural networks from overfitting.
#      JMLR 15, 1929-1958.
# [8]  Kingma, Ba (2015). Adam: A method for stochastic optimization. ICLR.
# [9]  Loshchilov, Hutter (2019). Decoupled weight decay regularization.
#      ICLR.  (AdamW)
# [10] Prechelt (1998). Early stopping, but when? In Neural Networks:
#      Tricks of the Trade, LNCS 1524, Springer, 55-69.
# [11] LeCun, Bottou, Orr, Mueller (1998). Efficient BackProp. In Neural
#      Networks: Tricks of the Trade, LNCS 1524, Springer, 9-50.
#      (input and target scaling)
# [12] Beyer, Hermans, Leibe (2015). Biternion nets: Continuous head pose
#      regression from discrete training labels. GCPR.  (predicting an
#      angle as a (cos, sin) pair)
# [13] Paszke et al. (2019). PyTorch: An imperative style, high-performance
#      deep learning library. NeurIPS.
#
# NEW DEPENDENCY
# ----------------
# torch (PyTorch). `pip install torch` -- on Windows the default PyPI wheel
# is CPU-only, which is all this file needs.
# =============================================================================

import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold

from splits import make_or_load_split, RANDOM_STATE
from forest_trainer_v3 import (
    load_table, score, is_circular, ALL_TARGETS, CV_FOLDS, _agg_stat,
)
import plot_predictions


RR_SCALE = 8000.0          # [m/s] fixed input scale for the range-rate channel
EARLY_STOP_FRACTION = 0.2  # share of the training data held back in stage 1
MAX_EPOCHS = 2000
PATIENCE = 150             # epochs without improvement before stopping
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

MODEL_NAME = "cnn"

torch.use_deterministic_algorithms(True)


# -----------------------------------------------------------------------------
# Data -> network input
# -----------------------------------------------------------------------------
def to_input(sub):
    """(N, 2, L) float32 array: channel 0 scaled range rate, channel 1 mask."""
    rr_cols = sorted(c for c in sub.columns if c.startswith("rr_c_"))
    mask_cols = sorted(c for c in sub.columns if c.startswith("mask_"))
    rr = sub[rr_cols].to_numpy(dtype=float) / RR_SCALE
    mask = sub[mask_cols].to_numpy(dtype=float)
    return np.stack([rr, mask], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# The network
# -----------------------------------------------------------------------------
class DopplerCNN(nn.Module):
    def __init__(self, seq_len, n_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat_len = 32 * (seq_len // 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(flat_len, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# One training run (used by both stages)
# -----------------------------------------------------------------------------
def _train(X, Y, n_out, seed, n_epochs, X_es=None, Y_es=None):
    """
    Train a fresh DopplerCNN on (X, Y) for up to n_epochs.
    If an early-stopping subset (X_es, Y_es) is given: evaluate it after
    every epoch, stop after PATIENCE epochs without improvement, restore the
    best weights, and return (model, best_epoch). Otherwise train for exactly
    n_epochs and return (model, n_epochs).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = DopplerCNN(X.shape[2], n_out)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    if X_es is not None:
        Xes_t, Yes_t = torch.from_numpy(X_es), torch.from_numpy(Y_es)

    best_loss, best_epoch, best_state, since_best = float("inf"), 0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        for start in range(0, len(Xt), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            optimiser.step()

        if X_es is None:
            continue

        model.eval()
        with torch.no_grad():
            es_loss = loss_fn(model(Xes_t), Yes_t).item()
        if es_loss < best_loss:
            best_loss, best_epoch, since_best = es_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break

    if X_es is None:
        return model, n_epochs
    model.load_state_dict(best_state)
    return model, best_epoch


# -----------------------------------------------------------------------------
# Regressor wrapper: target encoding + two-stage fit
# -----------------------------------------------------------------------------
class CNNRegressor:
    """
    fit(X, y) / predict(X), X shaped (N, 2, L), y in the label's own units
    (degrees for the circular labels).

    Linear targets: y standardised with the mean/std of the data being fit
    (never the test data), 1 output, un-standardised at predict time.
    Circular targets: y encoded as (sin, cos), 2 outputs, decoded with
    atan2 and wrapped to [0, 360) at predict time.
    """
    def __init__(self, circular, seed=RANDOM_STATE):
        self.circular = circular
        self.seed = seed

    def _encode(self, y):
        if self.circular:
            rad = np.radians(y)
            return np.stack([np.sin(rad), np.cos(rad)], axis=1).astype(np.float32)
        return ((y - self.y_mean_) / self.y_std_).astype(np.float32)[:, None]

    def _decode(self, out):
        if self.circular:
            return np.degrees(np.arctan2(out[:, 0], out[:, 1])) % 360.0
        return out[:, 0] * self.y_std_ + self.y_mean_

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        n_out = 2 if self.circular else 1

        # --- stage 1: find the epoch count on an early-stopping subset ---
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(X))
        n_es = int(round(EARLY_STOP_FRACTION * len(X)))
        es_idx, fit_idx = perm[:n_es], perm[n_es:]

        self.y_mean_, self.y_std_ = y[fit_idx].mean(), y[fit_idx].std()
        _, best_epoch = _train(
            X[fit_idx], self._encode(y[fit_idx]), n_out, self.seed, MAX_EPOCHS,
            X_es=X[es_idx], Y_es=self._encode(y[es_idx]),
        )
        self.best_epoch_ = max(best_epoch, 1)

        # --- stage 2: retrain on ALL the training data for that many epochs ---
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        self.model_, _ = _train(X, self._encode(y), n_out, self.seed, self.best_epoch_)
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X)).numpy()
        return self._decode(out)


# -----------------------------------------------------------------------------
# Held-out test and CV -- same structure and same folds as forest_trainer_v3
# -----------------------------------------------------------------------------
def held_out_test(df, target, train_ids, test_ids, seed=RANDOM_STATE):
    sub = df.dropna(subset=[target])
    n_dropped = len(df) - len(sub)
    train = sub[sub["sat_idx"].isin(train_ids)]
    test = sub[sub["sat_idx"].isin(test_ids)]

    model = CNNRegressor(circular=is_circular(target), seed=seed)
    model.fit(to_input(train), train[target].to_numpy())
    pred = model.predict(to_input(test))

    m = score(test[target].to_numpy(), pred, target)
    m.update({
        "target": target, "model_name": MODEL_NAME,
        "n_train": len(train), "n_test": len(test), "n_dropped_nan": n_dropped,
        "best_epoch": model.best_epoch_,
    })
    predictions = pd.DataFrame({
        "sat_idx": test["sat_idx"].to_numpy(), "target": target, "model": MODEL_NAME,
        "y_true": test[target].to_numpy(), "y_pred": pred,
    })
    return m, predictions


def cross_validate_in_train(df, target, train_ids, n_folds=CV_FOLDS, seed=RANDOM_STATE):
    """Identical fold construction to forest_trainer_v3.cross_validate_in_train
    (same row order, same KFold seed), so the CNN is scored on exactly the
    same 5 folds the trees were. Scored per fold, mean +/- std across folds."""
    train_only = df[df["sat_idx"].isin(train_ids)]
    sub = train_only.dropna(subset=[target]).reset_index(drop=True)
    n_dropped = len(train_only) - len(sub)

    X_all = to_input(sub)
    y_all = sub[target].to_numpy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores, fold_epochs = [], []
    for train_idx, val_idx in kf.split(X_all):
        model = CNNRegressor(circular=is_circular(target), seed=seed)
        model.fit(X_all[train_idx], y_all[train_idx])
        pred = model.predict(X_all[val_idx])
        fold_scores.append(score(y_all[val_idx], pred, target))
        fold_epochs.append(model.best_epoch_)

    m = {}
    for key in ("r2", "rmse", "circ_skill"):
        mean_v, std_v = _agg_stat([fs[key] for fs in fold_scores])
        m[key], m[f"{key}_std"] = mean_v, std_v
    m.update({
        "target": target, "model_name": MODEL_NAME,
        "n": len(sub), "n_dropped_nan": n_dropped, "fold_epochs": fold_epochs,
    })
    return m


if __name__ == "__main__":

    df, _ = load_table()
    split_df = make_or_load_split(df["sat_idx"])
    train_ids = set(split_df.loc[split_df["split"] == "train", "sat_idx"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "sat_idx"])

    L = sum(c.startswith("rr_c_") for c in df.columns)
    print(f"Loaded {len(df)} rows, input shape per satellite (2, {L}) "
          f"from output/training_table_padded.csv")
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test satellites "
          f"(output/train_test_split.csv, shared with forest_trainer_v3.py)")
    print(f"DopplerCNN trainable parameters: "
          f"{count_parameters(DopplerCNN(L, 1)):,} (linear labels), "
          f"{count_parameters(DopplerCNN(L, 2)):,} (circular labels)\n")

    t0 = time.time()

    test_results, test_predictions = [], []
    for target in ALL_TARGETS:
        m, preds = held_out_test(df, target, train_ids, test_ids)
        test_results.append(m)
        test_predictions.append(preds)

    print(f"{'target':>28} {'n_train':>8} {'n_test':>7} {'epochs':>7} "
          f"{'R2':>8} {'RMSE':>12} {'circ_skill':>11}")
    print("-" * 90)
    for r in test_results:
        r2_str = f"{r['r2']:.3f}" if not np.isnan(r["r2"]) else "  --  "
        skill_str = f"{r['circ_skill']:.3f}" if not np.isnan(r["circ_skill"]) else "  --  "
        print(f"{r['target']:>28} {r['n_train']:>8} {r['n_test']:>7} {r['best_epoch']:>7} "
              f"{r2_str:>8} {r['rmse']:>12.4f} {skill_str:>11}")

    print("\n" + "=" * 90)
    print("5-FOLD CROSS-VALIDATION ON THE TRAIN PARTITION ONLY (same folds as the trees)")
    print("=" * 90)

    cv_results = [cross_validate_in_train(df, target, train_ids) for target in ALL_TARGETS]

    print(f"\n{'target':>28} {'n':>5} {'R2 (+/-std)':>16} {'RMSE (+/-std)':>24} "
          f"{'circ_skill (+/-std)':>20}   fold epochs")
    print("-" * 120)
    for r in cv_results:
        r2_str = (f"{r['r2']:.3f}+/-{r['r2_std']:.3f}"
                  if not np.isnan(r["r2"]) else "  --  ")
        skill_str = (f"{r['circ_skill']:.3f}+/-{r['circ_skill_std']:.3f}"
                     if not np.isnan(r["circ_skill"]) else "  --  ")
        rmse_str = f"{r['rmse']:.4f}+/-{r['rmse_std']:.4f}"
        print(f"{r['target']:>28} {r['n']:>5} {r2_str:>16} {rmse_str:>24} "
              f"{skill_str:>20}   {r['fold_epochs']}")

    print(f"\nTotal training time: {time.time() - t0:.1f} s")

    # --- save, same schema as model_results_v3.csv so the two can be compared
    # or concatenated directly; separate files so neither run overwrites the other
    os.makedirs("output", exist_ok=True)

    pred_path = "output/test_predictions_cnn_v3.csv"
    pd.concat(test_predictions, ignore_index=True).to_csv(pred_path, index=False)

    out_rows = []
    for r in test_results:
        out_rows.append({
            "method": "single_split", "model": MODEL_NAME, "target": r["target"],
            "n": r["n_test"], "n_train": r["n_train"],
            "r2": r["r2"], "rmse": r["rmse"], "circ_skill": r["circ_skill"],
            "r2_std": np.nan, "rmse_std": np.nan, "circ_skill_std": np.nan,
        })
    for r in cv_results:
        out_rows.append({
            "method": "cv", "model": MODEL_NAME, "target": r["target"],
            "n": r["n"], "n_train": np.nan,
            "r2": r["r2"], "rmse": r["rmse"], "circ_skill": r["circ_skill"],
            "r2_std": r["r2_std"], "rmse_std": r["rmse_std"],
            "circ_skill_std": r["circ_skill_std"],
        })
    results_path = "output/model_results_cnn_v3.csv"
    pd.DataFrame(out_rows).to_csv(results_path, index=False)

    print(f"\nSaved held-out predictions to {pred_path}")
    print(f"Saved model results to {results_path}")

    plot_predictions.make_figure(
        predictions_path=pred_path, results_path=results_path,
        out_path="output/predicted_vs_true_cnn.png",
    )