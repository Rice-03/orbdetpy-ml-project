# =============================================================================
# splits.py -- shared 80/20 train/test split, keyed by sat_idx
# =============================================================================
#
# PURPOSE
# -------
# One fixed split, saved to disk, used by every model trained on this batch
# (LightGBM, Random Forest, and later the CNN). Per the project's stated
# evaluation method: 80/20 train/test, with cross-validation run on the
# training partition directly, not a separate reserved slice.
#
# Written once per batch; if the split file already exists, it is loaded
# as-is rather than regenerated, so a second trainer run (or a different
# model entirely) is scored against the same held-out satellites as the
# first. Delete output/train_test_split.csv to force a new split -- e.g.
# after the batch is regenerated (different sat_idx values), a stale split
# file would silently comment on the wrong satellites.
# =============================================================================

import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_or_load_split(sat_idx_values, split_path="output/train_test_split.csv",
                        test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    sat_idx_values : the sat_idx column of the table being split (order not
        important, duplicates not expected).

    Returns a DataFrame with columns [sat_idx, split] where split is
    "train" or "test". If split_path already exists, it is loaded and
    checked against sat_idx_values (every id must be covered) rather than
    regenerated -- this is what keeps a later CNN run on the exact same
    held-out set as this run.
    """
    sat_idx_values = sorted(set(int(s) for s in sat_idx_values))

    if os.path.exists(split_path):
        existing = pd.read_csv(split_path)
        existing_ids = set(existing["sat_idx"].astype(int))
        missing = set(sat_idx_values) - existing_ids
        if missing:
            raise SystemExit(
                f"{split_path} exists but is missing {len(missing)} sat_idx "
                f"values present in the current table (e.g. {sorted(missing)[:5]}). "
                "The split file is stale for this batch -- delete it and rerun "
                "to generate a fresh split, or confirm you're pointing at the "
                "batch it was built from."
            )
        return existing[existing["sat_idx"].isin(sat_idx_values)].reset_index(drop=True)

    train_ids, test_ids = train_test_split(
        sat_idx_values, test_size=test_size, random_state=random_state
    )
    rows = (
        [{"sat_idx": s, "split": "train"} for s in train_ids] +
        [{"sat_idx": s, "split": "test"} for s in test_ids]
    )
    split_df = pd.DataFrame(rows).sort_values("sat_idx").reset_index(drop=True)

    out_dir = os.path.dirname(split_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    split_df.to_csv(split_path, index=False)

    return split_df