"""Random forest baseline on the selected wearable features.

Usage:
    python random_forest.py <features_dir> <output_dir>

Reads FEATURES_{12H,24H,48H}.csv (made by preprocess_features.py) and, for each
labelling horizon, runs two tasks with participant-grouped cross-validation so
that no participant appears in both the training and the test fold:

  * multiclass: baseline / pre_ictal / attack / post_ictal / overlap
  * binary:     pre_ictal vs baseline (other states dropped)

Writes metrics, confusion matrices and feature importances to <output_dir>.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from preprocess_features import FEATURES, HORIZONS

N_SPLITS = 5
SEED = 42
FFILL_LIMIT = 6  # carry a reading forward at most 6 x 10 min = 1 hour
CLASS_ORDER = ["baseline", "pre_ictal", "attack", "post_ictal", "overlap"]


def load(features_dir: Path, horizon: str) -> pd.DataFrame:
    df = pd.read_csv(features_dir / f"FEATURES_{horizon}.csv")
    # "pre_ictal_12h" -> "pre_ictal", so classes share names across horizons.
    df["label"] = df["state_name"].str.replace(r"_\d+h$", "", regex=True)
    # Causal gap filling within each participant; remaining NaNs are handled
    # natively by scikit-learn's random forest.
    df[FEATURES] = df.groupby("participant_full_id")[FEATURES].ffill(limit=FFILL_LIMIT)
    return df


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def cross_validate(df: pd.DataFrame, classes: list[str]):
    X = df[FEATURES].to_numpy()
    y = df["label"].to_numpy()
    groups = df["participant_full_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(classes)))
    importances, fold_scores = [], []

    for train_idx, test_idx in cv.split(X, y, groups):
        model = make_model().fit(X[train_idx], y[train_idx])
        col = [list(model.classes_).index(c) if c in model.classes_ else None for c in classes]
        p = model.predict_proba(X[test_idx])
        for j, k in enumerate(col):
            if k is not None:
                proba[test_idx, j] = p[:, k]
        pred[test_idx] = model.predict(X[test_idx])
        importances.append(model.feature_importances_)
        fold_scores.append(f1_score(y[test_idx], pred[test_idx], labels=classes, average="macro", zero_division=0))

    return y, pred, proba, np.mean(importances, axis=0), fold_scores


def run_task(df: pd.DataFrame, classes: list[str], binary: bool) -> dict:
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    y, pred, proba, importances, fold_scores = cross_validate(df, classes)

    result = {
        "n_rows": int(len(y)),
        "class_counts": {c: int((y == c).sum()) for c in classes},
        "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
        "macro_f1_per_fold": fold_scores,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "report": classification_report(y, pred, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "feature_importance": dict(zip(FEATURES, importances.tolist())),
    }
    if binary:
        positive = (y == classes[1]).astype(int)
        result["roc_auc"] = roc_auc_score(positive, proba[:, 1])
        result["pr_auc"] = average_precision_score(positive, proba[:, 1])
        result["positive_rate"] = float(positive.mean())
    return result


def main(features_dir: str, output_dir: str) -> None:
    in_dir, out_dir = Path(features_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        "multiclass": (CLASS_ORDER, False),
        "binary_preictal_vs_baseline": (["baseline", "pre_ictal"], True),
    }
    results = {}
    for horizon in HORIZONS:
        df = load(in_dir, horizon)
        for task, (classes, binary) in tasks.items():
            r = run_task(df, classes, binary)
            results[f"{horizon}/{task}"] = r
            extra = f"  ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}" if binary else ""
            print(
                f"{horizon} {task}: macro-F1 {r['macro_f1']:.3f}  "
                f"balanced-acc {r['balanced_accuracy']:.3f}{extra}",
                flush=True,
            )
    (out_dir / "rf_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir / 'rf_results.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
