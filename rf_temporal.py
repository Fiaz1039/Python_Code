"""Random forest with per-person scaling and look-back features, evaluated
leave-one-participant-out.

Usage:
    python rf_temporal.py <features_dir> <output_dir> [horizon ...]

Feature sets compared (each adds to the previous one):
    raw       the 9 selected features as recorded
    scaled    each feature as a percentile within its own participant
    temporal  scaled + past-only rolling mean/std over 1h, 6h, 24h and the
              1h-vs-24h change
    time      temporal + time of day (sin/cos of hour), as a confound check

Tasks (negative class vs positive class):
    baseline vs pre_ictal, pre_ictal vs attack, attack vs post_ictal

Reliability: every fold holds out one whole participant. AUC is computed per
held-out participant and summarised as mean with a 95% bootstrap interval over
participants, as well as pooled over all rows.
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
    f1_score,
    recall_score,
    roc_auc_score,
)

from preprocess_features import FEATURES, HORIZONS
from random_forest import FFILL_LIMIT

SEED = 42
WINDOWS = ["1h", "6h", "24h"]
MIN_ATTACK_EPISODES = 2  # participants with 0 or 1 attack episodes are excluded
TASKS = {
    "baseline_vs_preictal": ("baseline", "pre_ictal"),
    "preictal_vs_attack": ("pre_ictal", "attack"),
    "attack_vs_postictal": ("attack", "post_ictal"),
}


def attack_episodes(df: pd.DataFrame) -> pd.Series:
    """Number of separate attack episodes (runs of consecutive attack rows) per participant."""
    is_attack = df["label"].eq("attack")
    starts = is_attack & ~is_attack.groupby(df["participant_full_id"]).shift(fill_value=False)
    return starts.groupby(df["participant_full_id"]).sum()


def load(features_dir: Path, horizon: str, min_attack_episodes: int = MIN_ATTACK_EPISODES) -> pd.DataFrame:
    df = pd.read_csv(features_dir / f"FEATURES_{horizon}.csv")
    df["label"] = df["state_name"].str.replace(r"_\d+h$", "", regex=True)
    df["time"] = pd.to_datetime(df["timestamp_unix"], unit="ms", utc=True)
    df = df.sort_values(["participant_full_id", "time"]).reset_index(drop=True)

    # Drop participants with too few attacks to evaluate on (none, or only one).
    episodes = attack_episodes(df)
    keep = episodes.index[episodes >= min_attack_episodes]
    dropped = sorted(set(episodes.index) - set(keep))
    if dropped:
        print(f"Excluding {len(dropped)} participants with < {min_attack_episodes} attack episodes: {dropped}")
    df = df[df["participant_full_id"].isin(keep)].reset_index(drop=True)

    df[FEATURES] = df.groupby("participant_full_id")[FEATURES].ffill(limit=FFILL_LIMIT)
    return df


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Add derived columns to df in place and return the column list per set."""
    by_person = df.groupby("participant_full_id")

    # Percentile within participant: robust to people whose EDA barely moves.
    scaled = [f"{f}_pct" for f in FEATURES]
    df[scaled] = by_person[FEATURES].rank(pct=True).to_numpy()

    # Past-only, time-based rolling windows (handle gaps in recording).
    temporal = []
    indexed = df.set_index("time").groupby("participant_full_id")[scaled]
    for w in WINDOWS:
        mean = indexed.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[[f"{c}_mean_{w}" for c in scaled]] = mean.to_numpy()
        temporal += [f"{c}_mean_{w}" for c in scaled]
        if w != "24h":
            std = indexed.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[[f"{c}_std_{w}" for c in scaled]] = std.to_numpy()
            temporal += [f"{c}_std_{w}" for c in scaled]
    for c in scaled:
        df[f"{c}_change"] = df[f"{c}_mean_1h"] - df[f"{c}_mean_24h"]
        temporal.append(f"{c}_change")

    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return {
        "raw": FEATURES,
        "scaled": scaled,
        "temporal": scaled + temporal,
        "time": scaled + temporal + ["hour_sin", "hour_cos"],
    }


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=20,
        max_samples=0.3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def bootstrap_ci(values: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def leave_one_participant_out(
    df: pd.DataFrame, columns: list[str], negative: str, positive: str, model_factory=None
) -> dict:
    model_factory = model_factory or make_model
    data = df[df["label"].isin([negative, positive])]
    X = data[columns].to_numpy()
    y = (data["label"] == positive).to_numpy().astype(int)
    pids = data["participant_full_id"].to_numpy()

    scores = np.zeros(len(y))
    per_person, importances = {}, []
    for pid in np.unique(pids):
        test = pids == pid
        if not test.any() or test.all():
            continue
        model = model_factory().fit(X[~test], y[~test])
        scores[test] = model.predict_proba(X[test])[:, 1]
        imp = np.asarray(model.feature_importances_, dtype=float)
        importances.append(imp / imp.sum())
        if 0 < y[test].sum() < test.sum():
            per_person[pid] = roc_auc_score(y[test], scores[test])

    aucs = np.array(list(per_person.values()))
    low, high = bootstrap_ci(aucs)
    importance = pd.Series(np.mean(importances, axis=0), index=columns).sort_values(ascending=False)
    return {
        "n_rows": {negative: int((y == 0).sum()), positive: int((y == 1).sum())},
        "pooled_roc_auc": roc_auc_score(y, scores),
        "pooled_pr_auc": average_precision_score(y, scores),
        "balanced_accuracy": balanced_accuracy_score(y, scores >= 0.5),
        "sensitivity": recall_score(y, scores >= 0.5),
        "specificity": recall_score(1 - y, scores < 0.5),
        "f1_positive": f1_score(y, scores >= 0.5),
        "positive_rate": float(y.mean()),
        "per_person_auc_mean": float(aucs.mean()),
        "per_person_auc_ci95": [low, high],
        "people_above_0.5": int((aucs > 0.5).sum()),
        "people_evaluated": int(len(aucs)),
        "per_person_auc": per_person,
        "top_features": importance.head(10).round(4).to_dict(),
    }


def main(features_dir: str, output_dir: str, horizons: list[str]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for horizon in horizons:
        df = load(Path(features_dir), horizon)
        feature_sets = build_feature_sets(df)
        for task, (negative, positive) in TASKS.items():
            for name, columns in feature_sets.items():
                r = leave_one_participant_out(df, columns, negative, positive)
                results[f"{horizon}/{task}/{name}"] = r
                lo, hi = r["per_person_auc_ci95"]
                print(
                    f"{horizon} {task:21s} {name:8s} n={r['n_rows']} "
                    f"per-person AUC {r['per_person_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}] "
                    f"({r['people_above_0.5']}/{r['people_evaluated']} >0.5)  "
                    f"pooled AUC {r['pooled_roc_auc']:.3f}  "
                    f"PR-AUC {r['pooled_pr_auc']:.3f} (base {r['positive_rate']:.3f})",
                    flush=True,
                )
        (out_dir / "rf_temporal_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir / 'rf_temporal_results.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or HORIZONS)
