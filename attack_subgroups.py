"""Do more painful or longer attacks classify better?

Usage:
    python attack_subgroups.py <features_dir> <diary.xlsx> <output_dir> [horizon]

Each sensor attack row is matched to its diary attack (same participant, time
inside the diary's UTC start/end) to get the attack's pain score (VAS 0-10) and
duration. Attacks are split at the median:
    pain:     high = VAS >= 6, low = VAS <= 5
    duration: long = duration >= median hours, short = below
Note: the diary's Intensita_del_dolore runs backwards (1 = most severe), so VAS
is used as the pain measure.

Attacks are compared with pre-ictal rows, leave-one-participant-out.

Design A (one model): a model is trained on all matched attacks vs pre-ictal.
  For each held-out patient, AUC is computed separately for their high and
  low subgroup attacks against the same pre-ictal rows. Same training data for
  both subgroups, so the comparison is fair. Patients with both subgroups are
  compared with a paired Wilcoxon test.
Design B (separate models): high attacks vs pre-ictal and low attacks vs
  pre-ictal are each trained and tested on their own.

Feature sets: temperature only, the duration set (activity counts, MET,
pulse, steps, EDA) and all 9 features.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from catboost_pairs import make_catboost
from preprocess_features import FEATURES
from rf_temporal import bootstrap_ci, leave_one_participant_out, load, make_model

NEGATIVE = "pre_ictal"
PAIN_THRESHOLD = 6
FEATURE_SETS = {
    "temperature_only": ["temperature_celsius"],
    "duration_set": ["activity_counts", "met", "pulse_rate_bpm", "step_counts", "eda_scl_usiemens"],
    "all_9": FEATURES,
}
MODELS = {"random_forest": make_model, "catboost": make_catboost}


def annotate(df: pd.DataFrame, diary_path: Path) -> tuple[pd.DataFrame, dict]:
    diary = pd.read_excel(diary_path)
    start = pd.to_datetime(diary["attack_start_utc"]).dt.tz_localize("UTC")
    end = pd.to_datetime(diary["attack_end_utc"]).dt.tz_localize("UTC")
    df = df.copy()
    df["vas"], df["duration_h"], df["attack_id"] = np.nan, np.nan, -1
    for i, row in diary.iterrows():
        m = (
            (df["participant_full_id"] == row["participant_full_id"])
            & (df["time"] >= start[i]) & (df["time"] < end[i])
            & (df["label"] == "attack")
        )
        df.loc[m, ["vas", "duration_h", "attack_id"]] = [row["Vas_dolore"], row["duration_hours"], i]

    unmatched = int(((df["label"] == "attack") & (df["attack_id"] < 0)).sum())
    df = df[(df["label"] != "attack") | (df["attack_id"] >= 0)].reset_index(drop=True)
    attacks = df[df["attack_id"] >= 0].groupby("attack_id")[["vas", "duration_h"]].first()
    duration_threshold = float(attacks["duration_h"].median())
    df["pain_group"] = np.where(df["vas"] >= PAIN_THRESHOLD, "high", "low")
    df["duration_group"] = np.where(df["duration_h"] >= duration_threshold, "high", "low")
    info = {
        "matched_attacks": int(len(attacks)),
        "unmatched_attack_rows_dropped": unmatched,
        "pain_threshold_vas": PAIN_THRESHOLD,
        "duration_threshold_hours": duration_threshold,
        "pain_high_attacks": int((attacks["vas"] >= PAIN_THRESHOLD).sum()),
        "pain_low_attacks": int((attacks["vas"] < PAIN_THRESHOLD).sum()),
        "duration_long_attacks": int((attacks["duration_h"] >= duration_threshold).sum()),
        "duration_short_attacks": int((attacks["duration_h"] < duration_threshold).sum()),
    }
    return df, info


def summarise_aucs(aucs: dict) -> dict:
    v = np.array(list(aucs.values()))
    lo, hi = bootstrap_ci(v)
    return {"auc": float(v.mean()), "ci95": [lo, hi], "above_0.5": f"{int((v > 0.5).sum())}/{len(v)}"}


def design_a(df: pd.DataFrame, group_col: str, columns: list[str], factory) -> dict:
    """One model on all attacks; AUC per held-out patient for each subgroup."""
    d = df[df["label"].isin([NEGATIVE, "attack"])].reset_index(drop=True)
    X = d[columns].to_numpy()
    y = (d["label"] == "attack").to_numpy().astype(int)
    pids = d["participant_full_id"].to_numpy()
    group = d[group_col].to_numpy()
    scores = np.zeros(len(y))
    for pid in np.unique(pids):
        test = pids == pid
        model = factory().fit(X[~test], y[~test])
        scores[test] = model.predict_proba(X[test])[:, 1]

    per_patient = {"high": {}, "low": {}}
    for pid in np.unique(pids):
        mine = pids == pid
        neg = mine & (y == 0)
        if not neg.any():
            continue
        for g in ("high", "low"):
            pos = mine & (y == 1) & (group == g)
            if pos.any():
                keep = neg | pos
                per_patient[g][pid] = roc_auc_score(y[keep], scores[keep])

    both = sorted(set(per_patient["high"]) & set(per_patient["low"]))
    diff = np.array([per_patient["high"][p] - per_patient["low"][p] for p in both])
    pooled = {}
    for g in ("high", "low"):
        keep = (y == 0) | (group == g)
        pooled[g] = roc_auc_score(y[keep], scores[keep])
    return {
        "high": summarise_aucs(per_patient["high"]),
        "low": summarise_aucs(per_patient["low"]),
        "pooled_auc_high": pooled["high"],
        "pooled_auc_low": pooled["low"],
        "paired_patients": len(both),
        "paired_high_minus_low": float(diff.mean()) if len(diff) else None,
        "paired_high_better": f"{int((diff > 0).sum())}/{len(diff)}",
        "paired_wilcoxon_p": float(wilcoxon(diff).pvalue) if len(diff) >= 2 else None,
        "per_patient": per_patient,
    }


def design_b(df: pd.DataFrame, group_col: str, columns: list[str], factory) -> dict:
    """Separate models: each subgroup's attacks vs pre-ictal."""
    out = {}
    for g in ("high", "low"):
        d = df[(df["label"] == NEGATIVE) | ((df["label"] == "attack") & (df[group_col] == g))].copy()
        r = leave_one_participant_out(d, columns, NEGATIVE, "attack", factory)
        lo, hi = r["per_person_auc_ci95"]
        out[g] = {
            "auc": r["per_person_auc_mean"], "ci95": [lo, hi],
            "above_0.5": f"{r['people_above_0.5']}/{r['people_evaluated']}",
            "pooled_auc": r["pooled_roc_auc"], "balanced_accuracy": r["balanced_accuracy"],
            "n_attack_rows": r["n_rows"]["attack"],
        }
    return out


def fmt(s: dict) -> str:
    return f"{s['auc']:.3f} [{s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}] ({s['above_0.5']})"


def main(features_dir: str, diary_path: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, info = annotate(load(Path(features_dir), horizon), Path(diary_path))
    print(json.dumps(info, indent=2), flush=True)
    results = {"info": info, "runs": []}
    for grouping, col in [("pain", "pain_group"), ("duration", "duration_group")]:
        for fs_name, columns in FEATURE_SETS.items():
            for model_name, factory in MODELS.items():
                a = design_a(df, col, columns, factory)
                b = design_b(df, col, columns, factory)
                results["runs"].append({
                    "grouping": grouping, "features": fs_name, "model": model_name,
                    "design_a_one_model": a, "design_b_separate_models": b,
                })
                p = a["paired_wilcoxon_p"]
                print(
                    f"{grouping:8s} {fs_name:16s} {model_name:13s} | A: high {fmt(a['high'])}  low {fmt(a['low'])}  "
                    f"paired diff {a['paired_high_minus_low']:+.3f} ({a['paired_high_better']} high better, "
                    f"p={p:.3f}) | B: high {fmt(b['high'])}  low {fmt(b['low'])}",
                    flush=True,
                )
                (out_dir / f"attack_subgroups_{horizon}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
