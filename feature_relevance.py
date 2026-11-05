"""Compare several feature-relevance techniques on the 9 selected features.

Usage:
    python feature_relevance.py <features_dir> <output_dir> [horizon]

Techniques (each answers a different question):
  1. Feature-feature Spearman correlation: which features are redundant?
  2. Pooled point-biserial correlation with the label: the naive "correlation
     with the class" over all rows of all people.
  3. Within-person effect size (Cohen's d per person, then averaged) with a
     Wilcoxon signed-rank test across people: does the feature move the same
     way in most people?
  4. Mutual information with the label (pooled): any dependency, not only
     linear.
  5. Permutation importance on unseen patients: how much held-out AUC drops
     when the feature is shuffled (participant-grouped 5-fold CV).
  6. Mean |SHAP| from CatBoost on unseen patients: how much the model's
     predictions lean on the feature.

Only 1-4 look at the data directly; 5 and 6 depend on a trained model.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import Pool  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from catboost_pairs import make_catboost  # noqa: E402
from feature_subset_search import SHORT, search_model  # noqa: E402
from preprocess_features import FEATURES  # noqa: E402
from rf_temporal import SEED, TASKS, load  # noqa: E402

N_SPLITS = 5
N_REPEATS = 5
MI_SAMPLE = 20000

# Chart tokens (light surface).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, NEUTRAL, RED = "#2a78d6", "#f0efec", "#e34948"
SERIES_1 = "#2a78d6"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def data_relevance(d: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = {}
    filled = d[FEATURES].fillna(d[FEATURES].median())
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d), size=min(MI_SAMPLE, len(d)), replace=False)
    mi = mutual_info_classif(filled.to_numpy()[idx], y[idx], random_state=SEED)
    for j, f in enumerate(FEATURES):
        x = filled[f].to_numpy()
        per_person = []
        for _, g in d.groupby("participant_full_id"):
            gy = y[g.index.to_numpy()]
            if 0 < gy.sum() < len(gy):
                per_person.append(cohens_d(g[f].to_numpy()[gy == 1], g[f].to_numpy()[gy == 0]))
        per_person = np.array([v for v in per_person if not np.isnan(v)])
        rows[f] = {
            "pooled_point_biserial_r": float(np.corrcoef(x, y)[0, 1]),
            "within_person_d_mean": float(per_person.mean()),
            "people_up": f"{int((per_person > 0).sum())}/{len(per_person)}",
            "wilcoxon_p": float(wilcoxon(per_person).pvalue),
            "mutual_information": float(mi[j]),
        }
    return pd.DataFrame(rows).T


def model_relevance(d: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    X = d[FEATURES].to_numpy()
    pids = d["participant_full_id"].to_numpy()
    rng = np.random.default_rng(SEED)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    drops = {f: [] for f in FEATURES}
    shap_abs = []
    for train_idx, test_idx in cv.split(X, y, pids):
        rf = search_model().set_params(n_jobs=-1).fit(X[train_idx], y[train_idx])
        X_test, y_test = X[test_idx], y[test_idx]
        base = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        for j, f in enumerate(FEATURES):
            for _ in range(N_REPEATS):
                X_perm = X_test.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                drops[f].append(base - roc_auc_score(y_test, rf.predict_proba(X_perm)[:, 1]))
        cb = make_catboost().fit(X[train_idx], y[train_idx])
        shap = cb.get_feature_importance(Pool(X_test), type="ShapValues")[:, :-1]
        shap_abs.append(np.abs(shap).mean(axis=0))
    shap_mean = np.mean(shap_abs, axis=0)
    return pd.DataFrame(
        {
            "perm_auc_drop_unseen": {f: float(np.mean(v)) for f, v in drops.items()},
            "perm_auc_drop_sd": {f: float(np.std(v)) for f, v in drops.items()},
            "catboost_mean_abs_shap_unseen": dict(zip(FEATURES, shap_mean.tolist())),
        }
    )


def plot_correlation(corr: pd.DataFrame, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    labels = [SHORT[f] for f in corr.columns]
    fig, ax = plt.subplots(figsize=(7.6, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", color=TEXT_SECONDARY, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    cbar.set_label("Spearman correlation", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Feature-feature correlation (all rows)", loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_permutation(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(13, 4.6), sharex=True, facecolor=SURFACE)
    for ax, (task, table) in zip(axes, results.items()):
        t = table.sort_values("perm_auc_drop_unseen")
        ypos = np.arange(len(t))
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.errorbar(
            t["perm_auc_drop_unseen"], ypos, xerr=t["perm_auc_drop_sd"], fmt="o", color=SERIES_1,
            ecolor=SERIES_1, elinewidth=2, capsize=0, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
        )
        ax.set_yticks(ypos, [SHORT[f] for f in t.index], color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(task.replace("_vs_", " vs ").replace("_", " "), loc="left", color=TEXT_PRIMARY, fontsize=11)
        ax.set_xlabel("AUC drop when shuffled (unseen patients)", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(
        "Permutation importance: right of 0 helps on new patients, left of 0 hurts",
        x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(features_dir: str, output_dir: str, horizon: str = "24H") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(features_dir), horizon)

    corr = df[FEATURES].corr(method="spearman")
    corr.to_csv(out_dir / "feature_correlation.csv")
    plot_correlation(corr, out_dir / "feature_correlation.png")

    results = {}
    for task, (negative, positive) in TASKS.items():
        d = df[df["label"].isin([negative, positive])].reset_index(drop=True)
        y = (d["label"] == positive).to_numpy().astype(int)
        table = data_relevance(d, y).join(model_relevance(d, y))
        table.to_csv(out_dir / f"relevance_{horizon}_{task}.csv")
        results[task] = table
        print(f"\n== {task} ({positive} = 1)", flush=True)
        print(table.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)

    plot_permutation(results, out_dir / "permutation_importance.png")
    (out_dir / f"relevance_{horizon}.json").write_text(
        json.dumps({t: tb.to_dict("index") for t, tb in results.items()}, indent=2)
    )
    print(f"\nSaved results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
