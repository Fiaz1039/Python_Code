"""CatBoost version of the pairwise comparisons in rf_temporal.py.

Usage:
    python catboost_pairs.py <features_dir> <output_dir> [horizon ...]

Uses the same features, class pairs and leave-one-participant-out evaluation
as rf_temporal.py, so the results compare directly with the random forest.
Defaults to the 24H labels.
"""

import json
import sys
from pathlib import Path

from catboost import CatBoostClassifier

from rf_temporal import SEED, TASKS, build_feature_sets, leave_one_participant_out, load


def make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
    )


def main(features_dir: str, output_dir: str, horizons: list[str]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for horizon in horizons:
        df = load(Path(features_dir), horizon)
        feature_sets = build_feature_sets(df)
        for task, (negative, positive) in TASKS.items():
            for name, columns in feature_sets.items():
                r = leave_one_participant_out(df, columns, negative, positive, make_catboost)
                results[f"{horizon}/{task}/{name}"] = r
                lo, hi = r["per_person_auc_ci95"]
                print(
                    f"{horizon} {task:21s} {name:8s} "
                    f"per-person AUC {r['per_person_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}] "
                    f"({r['people_above_0.5']}/{r['people_evaluated']} >0.5)  "
                    f"pooled AUC {r['pooled_roc_auc']:.3f}  "
                    f"PR-AUC {r['pooled_pr_auc']:.3f} (base {r['positive_rate']:.3f})",
                    flush=True,
                )
                (out_dir / "catboost_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir / 'catboost_results.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["24H"])
