"""Within-patient (personalised) models: each patient gets their own model,
trained and tested only on their own data.

Usage:
    python within_patient.py <features_dir> <output_dir> [horizon]

For every patient and class pair, 5-fold cross-validation is run on that
patient's rows only, with two fold types (the same as split_80_20.py --cv):
  random       stratified 5-fold over the patient's rows (neighbouring
               10-minute readings land in both train and test)
  time_ordered the patient's recording is cut into 5 consecutive blocks; each
               block is tested once, and training rows within PURGE_HOURS of
               the test block are dropped

Out-of-fold predictions give one AUC per patient. Results are summarised as
the mean over patients with a 95% bootstrap interval. Folds whose training
part holds only one class are skipped, and their rows left unscored.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost_pairs import make_catboost
from rf_temporal import SEED, TASKS, bootstrap_ci, load, make_model
from split_80_20 import FEATURE_SETS, N_FOLDS, time_blocked_folds

MODELS = {"random_forest": make_model, "catboost": make_catboost}


def random_folds_small(d: pd.DataFrame, y: np.ndarray):
    n_splits = min(N_FOLDS, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        return []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


FOLDS = {"random": random_folds_small, "time_ordered": time_blocked_folds}


def score_patient(X: np.ndarray, y: np.ndarray, folds, factory) -> np.ndarray:
    scores = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = factory().fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return scores


def main(features_dir: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(features_dir), horizon)
    summary, per_patient_rows = [], []

    for task, (negative, positive) in TASKS.items():
        pair = df[df["label"].isin([negative, positive])]
        for split_name, make_folds in FOLDS.items():
            for fs_name, columns in FEATURE_SETS.items():
                for model_name, factory in MODELS.items():
                    aucs, all_y, all_s = {}, [], []
                    for pid, d in pair.groupby("participant_full_id"):
                        d = d.reset_index(drop=True)
                        y = (d["label"] == positive).to_numpy().astype(int)
                        if not 0 < y.sum() < len(y):
                            continue
                        s = score_patient(d[columns].to_numpy(), y, make_folds(d, y), factory)
                        ok = ~np.isnan(s)
                        if ok.any() and 0 < y[ok].sum() < ok.sum():
                            aucs[pid] = roc_auc_score(y[ok], s[ok])
                            all_y.append(y[ok])
                            all_s.append(s[ok])
                            per_patient_rows.append({
                                "task": task, "split": split_name, "features": fs_name,
                                "model": model_name, "patient": pid, "auc": aucs[pid],
                                "n_scored": int(ok.sum()), "n_positive": int(y[ok].sum()),
                            })
                    values = np.array(list(aucs.values()))
                    lo, hi = bootstrap_ci(values)
                    y_all, s_all = np.concatenate(all_y), np.concatenate(all_s)
                    pred = s_all >= 0.5
                    r = {
                        "task": task, "split": split_name, "features": fs_name, "model": model_name,
                        "auc_mean": float(values.mean()), "auc_ci95": [lo, hi],
                        "patients_above_0.5": f"{int((values > 0.5).sum())}/{len(values)}",
                        "balanced_accuracy": balanced_accuracy_score(y_all, pred),
                        "sensitivity": recall_score(y_all, pred),
                        "specificity": recall_score(1 - y_all, ~pred),
                    }
                    summary.append(r)
                    print(
                        f"{task:21s} {split_name:12s} {fs_name:8s} {model_name:13s} "
                        f"AUC {r['auc_mean']:.3f} [{lo:.3f}-{hi:.3f}] ({r['patients_above_0.5']} >0.5)  "
                        f"bal-acc {r['balanced_accuracy']:.3f}  sens {r['sensitivity']:.3f}  "
                        f"spec {r['specificity']:.3f}",
                        flush=True,
                    )
                    (out_dir / f"within_patient_{horizon}.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(per_patient_rows).to_csv(out_dir / f"within_patient_{horizon}_per_patient.csv", index=False)
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
