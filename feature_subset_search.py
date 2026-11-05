"""Exhaustive search over subsets of the 9 selected features.

Usage:
    python feature_subset_search.py <features_dir> <output_dir> [horizon]

Stage 1 (search): for every non-empty subset of FEATURES (511) and every class
pair, a small random forest is evaluated with 5-fold participant-grouped
cross-validation. The score is the mean per-person ROC-AUC.

Stage 2 (summary):
  * ranking of subsets per pair
  * each feature's average effect: mean score of subsets that contain it
    minus mean score of subsets that do not
  * selection check: for each fold, the subset is chosen using only the
    other folds' people and scored on the held-out fold's people. This
    estimates how much of the "best subset" gain is real rather than luck
    from trying 511 combinations.

Stage 3 (confirm): the top 3 subsets per pair and the full 9-feature set are
re-evaluated leave-one-participant-out with the full random forest and CatBoost.
"""

import itertools
import json
import os
import sys
import multiprocessing
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from catboost_pairs import make_catboost  # noqa: E402
from preprocess_features import FEATURES  # noqa: E402
from rf_temporal import SEED, TASKS, leave_one_participant_out, load, make_model  # noqa: E402

N_SPLITS = 5
TOP_K = 3
SHORT = {
    "acticounts_magnitude": "acti_mag",
    "counts_magnitude": "counts_mag",
    "accelerometers_std_g": "acc_std",
    "activity_counts": "act_counts",
    "eda_scl_usiemens": "eda",
    "met": "met",
    "pulse_rate_bpm": "pulse",
    "step_counts": "steps",
    "temperature_celsius": "temp",
}

# Shared with worker processes (set before the pool forks).
_DATA: dict = {}


def search_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=60,
        min_samples_leaf=20,
        max_samples=0.3,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=SEED,
    )


def evaluate_subset(args) -> tuple[str, tuple[str, ...], dict]:
    task, subset = args
    X_all, y, pids, folds = _DATA[task]
    X = X_all[:, [FEATURES.index(f) for f in subset]]
    scores = np.zeros(len(y))
    for train_idx, test_idx in folds:
        model = search_model().fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    per_person = {}
    for pid in np.unique(pids):
        m = pids == pid
        if 0 < y[m].sum() < m.sum():
            per_person[pid] = roc_auc_score(y[m], scores[m])
    return task, subset, per_person


def prepare(df: pd.DataFrame) -> dict:
    data = {}
    for task, (negative, positive) in TASKS.items():
        d = df[df["label"].isin([negative, positive])]
        X = d[FEATURES].to_numpy()
        y = (d["label"] == positive).to_numpy().astype(int)
        pids = d["participant_full_id"].to_numpy()
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(cv.split(X, y, pids))
        data[task] = (X, y, pids, folds)
    return data


def subset_name(subset) -> str:
    return " + ".join(SHORT[f] for f in subset)


def summarise(task: str, results: dict, fold_people: list[set]) -> dict:
    rows = []
    for subset, per_person in results.items():
        rows.append({"subset": subset, "n_features": len(subset), "auc": np.mean(list(per_person.values()))})
    table = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    full = tuple(FEATURES)

    # Average effect of including each feature.
    effect = {}
    for f in FEATURES:
        has = table["subset"].apply(lambda s: f in s)
        effect[f] = float(table.loc[has, "auc"].mean() - table.loc[~has, "auc"].mean())

    # Best subset for each size.
    best_by_size = (
        table.loc[table.groupby("n_features")["auc"].idxmax()]
        .sort_values("n_features")
        .assign(subset=lambda t: t["subset"].apply(subset_name))
    )

    # Selection check: choose using other folds' people, score on held-out people.
    chosen, selected_scores, full_scores = [], [], []
    for people in fold_people:
        def mean_auc(subset, keep):
            vals = [a for p, a in results[subset].items() if (p in people) == keep]
            return np.mean(vals) if vals else np.nan

        best = max(results, key=lambda s: mean_auc(s, keep=False))
        chosen.append(subset_name(best))
        selected_scores.append(mean_auc(best, keep=True))
        full_scores.append(mean_auc(full, keep=True))

    return {
        "table": table,
        "full_auc": float(table.loc[table["subset"] == full, "auc"].iloc[0]),
        "full_rank": int(table.index[table["subset"] == full][0]) + 1,
        "feature_effect": dict(sorted(effect.items(), key=lambda kv: -kv[1])),
        "best_by_size": best_by_size[["n_features", "subset", "auc"]].to_dict("records"),
        "selection_check": {
            "subsets_chosen_per_fold": chosen,
            "selected_auc_on_heldout": float(np.nanmean(selected_scores)),
            "full_set_auc_on_heldout": float(np.nanmean(full_scores)),
        },
    }


def main(features_dir: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(features_dir), horizon)
    _DATA.update(prepare(df))

    subsets = [s for k in range(1, len(FEATURES) + 1) for s in itertools.combinations(FEATURES, k)]
    jobs = [(task, s) for task in TASKS for s in subsets]
    print(f"Stage 1: {len(subsets)} subsets x {len(TASKS)} pairs = {len(jobs)} evaluations", flush=True)

    results = {task: {} for task in TASKS}
    with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
        for i, (task, subset, per_person) in enumerate(pool.imap_unordered(evaluate_subset, jobs, chunksize=4), 1):
            results[task][subset] = per_person
            if i % 150 == 0:
                print(f"  {i}/{len(jobs)} done", flush=True)

    raw = {task: {subset_name(s): pp for s, pp in r.items()} for task, r in results.items()}
    (out_dir / f"subset_search_{horizon}.json").write_text(json.dumps(raw, indent=2))

    summary = {}
    for task in TASKS:
        _, _, pids, folds = _DATA[task]
        fold_people = [set(pids[test_idx]) for _, test_idx in folds]
        s = summarise(task, results[task], fold_people)
        table = s.pop("table")
        table.assign(subset=table["subset"].apply(subset_name)).to_csv(
            out_dir / f"subset_ranking_{horizon}_{task}.csv", index=False
        )
        s["top"] = [
            {"subset": subset_name(r.subset), "n_features": r.n_features, "auc": r.auc}
            for r in table.head(10).itertuples()
        ]
        s["top_subsets_raw"] = [list(x) for x in table["subset"].head(TOP_K)]
        summary[task] = s
        sel = s["selection_check"]
        print(
            f"\n== {task}: full set AUC {s['full_auc']:.3f} (rank {s['full_rank']}/511); "
            f"held-out check: selected {sel['selected_auc_on_heldout']:.3f} vs full "
            f"{sel['full_set_auc_on_heldout']:.3f}",
            flush=True,
        )
        for r in s["top"][:5]:
            print(f"   {r['auc']:.3f}  {r['subset']}", flush=True)

    print("\nStage 3: confirming top subsets leave-one-participant-out", flush=True)
    models = {"random_forest": make_model, "catboost": make_catboost}
    for task, (negative, positive) in TASKS.items():
        candidates = summary[task]["top_subsets_raw"] + [list(FEATURES)]
        confirm = []
        for subset in candidates:
            for model_name, factory in models.items():
                r = leave_one_participant_out(df, subset, negative, positive, factory)
                confirm.append({
                    "subset": "all 9 features" if len(subset) == len(FEATURES) else subset_name(subset),
                    "model": model_name,
                    "auc": r["per_person_auc_mean"],
                    "ci95": r["per_person_auc_ci95"],
                    "people_above_0.5": f"{r['people_above_0.5']}/{r['people_evaluated']}",
                    "balanced_accuracy": r["balanced_accuracy"],
                    "sensitivity": r["sensitivity"],
                    "specificity": r["specificity"],
                })
                c = confirm[-1]
                print(
                    f"  {task:21s} {model_name:13s} AUC {c['auc']:.3f} "
                    f"[{c['ci95'][0]:.3f}-{c['ci95'][1]:.3f}] ({c['people_above_0.5']})  "
                    f"bal-acc {c['balanced_accuracy']:.3f}  {c['subset']}",
                    flush=True,
                )
        summary[task]["confirmation_loso"] = confirm
        (out_dir / f"subset_summary_{horizon}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
