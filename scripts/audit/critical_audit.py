"""
Critical audit of the model — what does it ACTUALLY use, what is noise,
what is leaking, and where does it break?

Outputs human-readable findings to reports/AUDIT.md and figures.

Inspections:
  1. Feature importance distributions (low-importance tail = candidate for removal).
  2. Pairwise correlation matrix → groups of duplicates.
  3. Per-feature NaN ratio.
  4. Per-fold MAPE → which test year breaks the model?
  5. Per-field residual analysis → does the model just memorize field means?
  6. Crop-specific quirks: separate sunflower / wheat_spring / wheat_winter.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
REPORTS = Path("reports")
FIGS = REPORTS / "figures" / "audit"
FIGS.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id", "rotation_pair"]
NON_FEATURE = ["field_id", "year", "target_yield_t_ha", "target_source", "unit_fix_applied", "standard_name"]
WALK_FORWARD_TEST_YEARS = list(range(2017, 2026))

CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


def prepare_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    feat_cols = list(dict.fromkeys(feat_cols + ["field_id"]))
    X = df[feat_cols].copy()
    cat_in_X = [c for c in CAT_FEATURES if c in X.columns]
    for c in cat_in_X:
        if c == "rotation_pair":
            X[c] = X[c].fillna("missing").astype(str)
        else:
            X[c] = X[c].apply(lambda v: "missing" if pd.isna(v) else str(int(float(v))))
    for c in [c for c in feat_cols if c not in cat_in_X]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, df["target_yield_t_ha"].astype(float), feat_cols, cat_in_X


def fit_full(df: pd.DataFrame):
    X, y, feat_cols, cat_in_X = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in cat_in_X]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=5, l2_leaf_reg=3.0,
        random_seed=42, verbose=False, allow_writing_files=False,
    )
    m.fit(Pool(X, y, cat_features=cat_idx))
    return m, X, y, feat_cols


def walk_forward_per_fold(df: pd.DataFrame):
    X, y, feat_cols, cat_in_X = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in cat_in_X]
    years = df["year"].values
    fold_rows = []
    oof = np.full(len(df), np.nan)
    for ty in WALK_FORWARD_TEST_YEARS:
        test_mask = years == ty
        train_mask = years < ty
        if test_mask.sum() == 0 or train_mask.sum() < 30 or len(pd.unique(years[train_mask])) < 4:
            continue
        Xtr, ytr = X.iloc[train_mask], y.iloc[train_mask]
        Xte, yte = X.iloc[test_mask], y.iloc[test_mask]
        n = len(Xtr)
        rng = np.random.default_rng(42)
        idx = np.arange(n); rng.shuffle(idx)
        v = max(5, int(n * 0.2))
        val_idx, tr_idx = idx[:v], idx[v:]
        m = CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=5, l2_leaf_reg=3.0,
            random_seed=42, verbose=False, allow_writing_files=False,
            od_type="Iter", od_wait=50,
        )
        m.fit(
            Pool(Xtr.iloc[tr_idx], ytr.iloc[tr_idx], cat_features=cat_idx),
            eval_set=Pool(Xtr.iloc[val_idx], ytr.iloc[val_idx], cat_features=cat_idx),
        )
        pred = m.predict(Xte)
        oof[test_mask] = pred
        # naive baseline = mean of train
        naive_pred = float(ytr.mean())
        # per-field-mean baseline
        field_mean_map = df.iloc[train_mask].groupby("field_id")["target_yield_t_ha"].mean().to_dict()
        field_pred = df.iloc[test_mask]["field_id"].map(field_mean_map).fillna(naive_pred).values
        fold_rows.append({
            "test_year": int(ty),
            "n_train": int(n), "n_test": int(test_mask.sum()),
            "n_train_years": int(len(pd.unique(years[train_mask]))),
            "ml_mape": float(np.mean(np.abs((yte - pred) / np.where(yte > 0, yte, np.nan))) * 100),
            "ml_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "ml_r2": float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan"),
            "naive_mape": float(np.mean(np.abs((yte - naive_pred) / np.where(yte > 0, yte, np.nan))) * 100),
            "field_mean_mape": float(np.mean(np.abs((yte - field_pred) / np.where(yte > 0, yte, np.nan))) * 100),
        })
    return pd.DataFrame(fold_rows), oof


def correlations(df: pd.DataFrame, target: str = "target_yield_t_ha") -> pd.DataFrame:
    num = df.select_dtypes(include=["number"]).copy()
    if target not in num.columns:
        return pd.DataFrame()
    cor = num.corr(method="spearman")[[target]].rename(columns={target: "corr_target"})
    return cor.dropna().sort_values("corr_target", ascending=False)


def find_redundant_pairs(df: pd.DataFrame, threshold: float = 0.95) -> list[tuple[str, str, float]]:
    num = df.select_dtypes(include=["number"]).drop(columns=["target_yield_t_ha", "year", "field_id", "crop_id", "prev_crop_id"], errors="ignore")
    cor = num.corr(method="spearman").abs()
    pairs = []
    cols = cor.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cor.loc[a, b]
            if pd.notna(v) and v >= threshold:
                pairs.append((a, b, float(v)))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def audit_scenario(scenario: str, crop_filter: list[str] | None, tag: str) -> dict:
    path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{tag}.csv"
    df = pd.read_csv(path).merge(CROPS_DF, on="crop_id", how="left")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
    res = {"scenario": scenario, "tag": tag, "n_rows": len(df), "n_features": df.shape[1]}

    # ---- 1. NaN ratio per feature
    nan_ratio = df.isna().mean().sort_values(ascending=False)
    res["high_nan"] = nan_ratio[nan_ratio > 0.5].to_dict()

    # ---- 2. correlation with target
    cor = correlations(df)
    res["top_pos_corr"] = cor.head(10).round(3).to_dict()["corr_target"]
    res["top_neg_corr"] = cor.tail(10).round(3).to_dict()["corr_target"]

    # ---- 3. redundant pairs
    res["redundant_pairs_top10"] = find_redundant_pairs(df, threshold=0.95)[:15]

    # ---- 4. feature importance from full-data fit
    m, X, y, feat_cols = fit_full(df)
    imp = pd.DataFrame({"feature": feat_cols, "importance": m.get_feature_importance()})
    imp = imp.sort_values("importance", ascending=False)
    res["importance_top15"] = imp.head(15).round(3).to_dict("records")
    res["importance_bottom15"] = imp.tail(15).round(3).to_dict("records")

    # ---- 5. per-fold metrics + naive baselines
    fold, oof = walk_forward_per_fold(df)
    res["fold_metrics"] = fold.round(3).to_dict("records")
    if not fold.empty:
        res["mean_ml_mape"] = float(fold["ml_mape"].mean())
        res["mean_naive_mape"] = float(fold["naive_mape"].mean())
        res["mean_field_mean_mape"] = float(fold["field_mean_mape"].mean())

    # ---- 6. per-field residual concentration
    df_oof = df.assign(oof=oof, residual=df["target_yield_t_ha"] - oof)
    res["field_residual_std"] = (
        df_oof.dropna(subset=["oof"])
        .groupby("field_id")["residual"]
        .std()
        .describe()
        .round(3)
        .to_dict()
    )

    # ---- 7. plot fold MAPE
    if not fold.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fold["test_year"], fold["ml_mape"], marker="o", label="ML")
        ax.plot(fold["test_year"], fold["naive_mape"], marker="s", linestyle="--", label="Naive (global mean)")
        ax.plot(fold["test_year"], fold["field_mean_mape"], marker="^", linestyle=":", label="Naive (field mean)")
        ax.set_title(f"{scenario} as-of {tag} — per-fold MAPE")
        ax.set_xlabel("Test year"); ax.set_ylabel("MAPE %"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        p = FIGS / f"fold_mape_{scenario}_{tag}.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        res["fold_plot"] = p.name

    return res


def write_md(all_res: list[dict]):
    out = ["# Critical Model Audit (v4)\n\n"]
    out.append("Generated by `scripts/audit/critical_audit.py`. Goal: find feature noise,\n")
    out.append("redundancy, leakage and per-fold instability.\n\n")
    for r in all_res:
        out.append(f"## {r['scenario']} — as-of {r['tag']}\n\n")
        out.append(f"- rows: {r['n_rows']}, features: {r['n_features']}\n")
        if "mean_ml_mape" in r:
            out.append(f"- mean ML MAPE: {r['mean_ml_mape']:.2f}%, naive (global mean): {r['mean_naive_mape']:.2f}%, naive (field mean): {r['mean_field_mean_mape']:.2f}%\n")
            if r["mean_ml_mape"] >= r["mean_naive_mape"]:
                out.append("  - ⚠️ **ML does NOT beat naive global mean** for this scenario.\n")
            if r["mean_ml_mape"] >= r["mean_field_mean_mape"]:
                out.append("  - ⚠️ **ML does NOT beat per-field mean baseline** — model adds noise on top of field memorization.\n")
        out.append("\n### Per-fold metrics\n\n")
        out.append("| year | n_test | n_train | n_train_years | ML MAPE | naive | field_mean | ML R² |\n|---|---|---|---|---|---|---|---|\n")
        for f in r.get("fold_metrics", []):
            out.append(f"| {f['test_year']} | {f['n_test']} | {f['n_train']} | {f['n_train_years']} | {f['ml_mape']:.1f} | {f['naive_mape']:.1f} | {f['field_mean_mape']:.1f} | {f['ml_r2']:.2f} |\n")
        out.append("\n### Top features by importance\n\n")
        for fr in r.get("importance_top15", []):
            out.append(f"- `{fr['feature']}` — {fr['importance']:.2f}\n")
        out.append("\n### Bottom 15 features (candidates for removal)\n\n")
        for fr in r.get("importance_bottom15", []):
            out.append(f"- `{fr['feature']}` — {fr['importance']:.2f}\n")
        out.append("\n### Redundant pairs (Spearman ≥ 0.95)\n\n")
        for a, b, v in r.get("redundant_pairs_top10", []):
            out.append(f"- `{a}` ↔ `{b}`: ρ = {v:.3f}\n")
        out.append("\n### Top |corr| with target\n\n")
        for k, v in r.get("top_pos_corr", {}).items():
            out.append(f"- `{k}`: {v:+.3f}\n")
        out.append("...\n")
        for k, v in r.get("top_neg_corr", {}).items():
            out.append(f"- `{k}`: {v:+.3f}\n")
        out.append("\n### High-NaN features\n\n")
        for k, v in r.get("high_nan", {}).items():
            out.append(f"- `{k}`: {v*100:.1f}%\n")
        if "fold_plot" in r:
            out.append(f"\n![fold mape](figures/audit/{r['fold_plot']})\n\n")
        out.append("\n---\n\n")
    (REPORTS / "AUDIT.md").write_text("".join(out), encoding="utf-8")


def main():
    print("=" * 80)
    print("CRITICAL AUDIT")
    print("=" * 80)
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_spring", ["wheat_spring"]),
        ("wheat_winter", ["wheat_winter"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    tags = ["07_01", "08_01", "09_01"]
    all_res = []
    for sc, cf in scenarios:
        for tag in tags:
            try:
                print(f"\n--- {sc} {tag} ---")
                r = audit_scenario(sc, cf, tag)
                all_res.append(r)
                if "mean_ml_mape" in r:
                    print(f"  rows={r['n_rows']}  ML={r['mean_ml_mape']:.2f}%  naive_global={r['mean_naive_mape']:.2f}%  naive_field_mean={r['mean_field_mean_mape']:.2f}%")
                else:
                    print(f"  rows={r['n_rows']}  no folds (too small)")
            except Exception as e:
                print(f"  ERROR: {e}")
    write_md(all_res)
    print(f"\nSaved {REPORTS / 'AUDIT.md'}")


if __name__ == "__main__":
    main()
