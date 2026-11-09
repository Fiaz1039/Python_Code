"""Run the three class pairs on attack subgroups (pain, duration) with all 9 features.

Usage:
    python subgroup_pairs.py <features_dir> <diary.xlsx> <output_dir> [horizon]

Each attack is grouped by its diary record (see attack_subgroups.py):
    pain:     high = VAS >= 6, low = VAS <= 5
    duration: long = duration >= median hours, short = below
Pre-ictal rows are assigned to the next attack (if it starts within the
labelling window) and post-ictal rows to the previous attack (if it ended
within the window), so each subgroup keeps its own attacks together with
their pre- and post-ictal periods. Baseline rows are kept in every subgroup;
overlap rows and rows linked to attacks missing from the diary are dropped.

For every subgroup the pairs baseline/pre-ictal, pre-ictal/attack and
attack/post-ictal are evaluated leave-one-participant-out with a random
forest and CatBoost on all 9 features.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from attack_subgroups import annotate
from catboost_pairs import make_catboost
from preprocess_features import FEATURES
from rf_temporal import TASKS, leave_one_participant_out, load, make_model

MODELS = {"random_forest": make_model, "catboost": make_catboost}
SLACK = pd.Timedelta(minutes=10)


def link_peri_ictal(df: pd.DataFrame, window: pd.Timedelta) -> pd.DataFrame:
    """Copy each attack's pain/duration group onto its pre- and post-ictal rows."""
    df = df.copy()
    for col in ("pain_group", "duration_group"):
        df.loc[df["attack_id"] < 0, col] = None
    for pid, g in df.groupby("participant_full_id"):
        att = g[g["attack_id"] >= 0].groupby("attack_id").agg(
            start=("time", "min"), end=("time", "max"),
            pain=("pain_group", "first"), duration=("duration_group", "first"),
        ).sort_values("start")
        if att.empty:
            continue
        starts, ends = att["start"].to_numpy(), att["end"].to_numpy()

        pre = g.index[g["label"] == "pre_ictal"]
        t = g.loc[pre, "time"].to_numpy()
        nxt = np.searchsorted(starts, t, side="right")
        ok = (nxt < len(att)) & ((starts[np.minimum(nxt, len(att) - 1)] - t) <= (window + SLACK).to_timedelta64())
        df.loc[pre[ok], "pain_group"] = att["pain"].to_numpy()[nxt[ok]]
        df.loc[pre[ok], "duration_group"] = att["duration"].to_numpy()[nxt[ok]]

        post = g.index[g["label"] == "post_ictal"]
        t = g.loc[post, "time"].to_numpy()
        prv = np.searchsorted(ends, t, side="left") - 1
        ok = (prv >= 0) & ((t - ends[np.maximum(prv, 0)]) <= (window + SLACK).to_timedelta64())
        df.loc[post[ok], "pain_group"] = att["pain"].to_numpy()[prv[ok]]
        df.loc[post[ok], "duration_group"] = att["duration"].to_numpy()[prv[ok]]
    return df


SUBGROUPS = {
    "all_attacks": (None, None),
    "pain_high": ("pain_group", "high"),
    "pain_low": ("pain_group", "low"),
    "duration_long": ("duration_group", "high"),
    "duration_short": ("duration_group", "low"),
}


def subset(df: pd.DataFrame, col, value) -> pd.DataFrame:
    peri = df["label"].isin(["pre_ictal", "attack", "post_ictal"])
    if col is None:
        linked = df["pain_group"].notna()
    else:
        linked = df[col] == value
    return df[(df["label"] == "baseline") | (peri & linked)].reset_index(drop=True)


def main(features_dir: str, diary_path: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = pd.Timedelta(hours=int(horizon.rstrip("H")))
    df, info = annotate(load(Path(features_dir), horizon), Path(diary_path))
    df = link_peri_ictal(df, window)

    results = {"info": info, "runs": []}
    for name, (col, value) in SUBGROUPS.items():
        d = subset(df, col, value)
        counts = d["label"].value_counts().to_dict()
        print(f"\n== {name}: rows {counts}", flush=True)
        for task, (negative, positive) in TASKS.items():
            for model_name, factory in MODELS.items():
                r = leave_one_participant_out(d, FEATURES, negative, positive, factory)
                lo, hi = r["per_person_auc_ci95"]
                results["runs"].append({
                    "subgroup": name, "task": task, "model": model_name,
                    "n_rows": r["n_rows"], "auc": r["per_person_auc_mean"], "ci95": [lo, hi],
                    "above_0.5": f"{r['people_above_0.5']}/{r['people_evaluated']}",
                    "pooled_auc": r["pooled_roc_auc"], "balanced_accuracy": r["balanced_accuracy"],
                    "sensitivity": r["sensitivity"], "specificity": r["specificity"],
                    "per_person_auc": r["per_person_auc"],
                })
                print(
                    f"   {task:21s} {model_name:13s} AUC {r['per_person_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}] "
                    f"({r['people_above_0.5']}/{r['people_evaluated']})  bal-acc {r['balanced_accuracy']:.3f}  "
                    f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  n={r['n_rows']}",
                    flush=True,
                )
                (out_dir / f"subgroup_pairs_{horizon}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
