"""80/20 train/test split where the same patients appear in training and test.

Usage:
    python split_80_20.py <features_dir> <output_dir> [horizon]

Two splits are compared:
  random       80/20 random split of all rows, stratified by class. Rows from
               the same patient (often 10 minutes apart) land in both sets.
  time_ordered within each patient, the first 80% of their recording is
               training and the last 20% is test. Patients are still seen in
               training, but test rows are not interleaved with training rows.

Models: random forest and CatBoost. Feature sets: EDA + temperature, all 9.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from catboost_pairs import make_catboost
from preprocess_features import FEATURES
from rf_temporal import SEED, TASKS, load, make_model

TEST_SIZE = 0.2
FEATURE_SETS = {
    "eda_temp": ["eda_scl_usiemens", "temperature_celsius"],
    "all_9": FEATURES,
}
MODELS = {"random_forest": make_model, "catboost": make_catboost}


def random_split(d: pd.DataFrame, y: np.ndarray):
    idx = np.arange(len(d))
    return train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def time_ordered_split(d: pd.DataFrame, y: np.ndarray):
    position = d.groupby("participant_full_id").cumcount()
    size = d.groupby("participant_full_id")["participant_full_id"].transform("size")
    is_test = (position >= np.floor(size * (1 - TEST_SIZE))).to_numpy()
    idx = np.arange(len(d))
    return idx[~is_test], idx[is_test]


SPLITS = {"random": random_split, "time_ordered": time_ordered_split}


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    pred = score >= 0.5
    return {
        "roc_auc": roc_auc_score(y_true, score),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred),
        "specificity": recall_score(1 - y_true, ~pred),
        "f1_positive": f1_score(y_true, pred),
    }


def main(features_dir: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(features_dir), horizon)
    results = []
    for task, (negative, positive) in TASKS.items():
        # load() returns rows sorted by participant then time, which the
        # time-ordered split relies on.
        d = df[df["label"].isin([negative, positive])].reset_index(drop=True)
        y = (d["label"] == positive).to_numpy().astype(int)
        for split_name, split in SPLITS.items():
            train_idx, test_idx = split(d, y)
            for fs_name, columns in FEATURE_SETS.items():
                X = d[columns].to_numpy()
                for model_name, factory in MODELS.items():
                    model = factory().fit(X[train_idx], y[train_idx])
                    score = model.predict_proba(X[test_idx])[:, 1]
                    r = {
                        "task": task, "split": split_name, "features": fs_name, "model": model_name,
                        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                        "test_positive_rate": float(y[test_idx].mean()),
                        **evaluate(y[test_idx], score),
                    }
                    results.append(r)
                    print(
                        f"{task:21s} {split_name:12s} {fs_name:8s} {model_name:13s} "
                        f"AUC {r['roc_auc']:.3f}  acc {r['accuracy']:.3f}  "
                        f"bal-acc {r['balanced_accuracy']:.3f}  sens {r['sensitivity']:.3f}  "
                        f"spec {r['specificity']:.3f}  F1 {r['f1_positive']:.3f}",
                        flush=True,
                    )
    pd.DataFrame(results).to_csv(out_dir / f"split_80_20_{horizon}.csv", index=False)
    (out_dir / f"split_80_20_{horizon}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
