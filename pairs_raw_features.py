"""Pairwise comparisons using only the 9 selected features, with no derived
columns (no per-person scaling, look-back windows or time of day).

Usage:
    python pairs_raw_features.py <features_dir> <output_dir> [horizon ...] [--features a,b,...]

--features limits the run to a comma-separated subset of the 9 features.

Runs a random forest and CatBoost on each class pair, evaluated
leave-one-participant-out. Defaults to the 24H labels.
Metrics at the 0.5 threshold use class-balanced models.
"""

import json
import sys
from pathlib import Path

from catboost_pairs import make_catboost
from preprocess_features import FEATURES
from rf_temporal import TASKS, leave_one_participant_out, load, make_model

MODELS = {"random_forest": make_model, "catboost": make_catboost}


def main(features_dir: str, output_dir: str, horizons: list[str], features: list[str] = FEATURES) -> None:
    unknown = set(features) - set(FEATURES)
    if unknown:
        sys.exit(f"Unknown features: {sorted(unknown)}")
    tag = "all" if list(features) == FEATURES else "_".join(features)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for horizon in horizons:
        df = load(Path(features_dir), horizon)
        for task, (negative, positive) in TASKS.items():
            for model_name, factory in MODELS.items():
                r = leave_one_participant_out(df, list(features), negative, positive, factory)
                results[f"{horizon}/{task}/{model_name}"] = r
                lo, hi = r["per_person_auc_ci95"]
                print(
                    f"{horizon} {task:21s} {model_name:13s} "
                    f"per-person AUC {r['per_person_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}] "
                    f"({r['people_above_0.5']}/{r['people_evaluated']} >0.5)  "
                    f"pooled AUC {r['pooled_roc_auc']:.3f}  "
                    f"bal-acc {r['balanced_accuracy']:.3f}  "
                    f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  "
                    f"F1 {r['f1_positive']:.3f}",
                    flush=True,
                )
                (out_dir / f"raw_features_results_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir / f'raw_features_results_{tag}.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = sys.argv[3:]
    chosen = FEATURES
    if "--features" in args:
        i = args.index("--features")
        chosen = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    main(sys.argv[1], sys.argv[2], args or ["24H"], chosen)
