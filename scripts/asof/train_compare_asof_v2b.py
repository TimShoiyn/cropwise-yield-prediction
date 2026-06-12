"""
Sprint 3.2 — Train as-of CatBoost on v2b (with agronomic features) and compare:
  - v2  (Sprint 3.1 features) vs v2b  (Sprint 3.2 enriched features)
  - v2b vs Cropwise own forecast

Outputs:
  - models_v2/asof_comparison/asof_results_v2b.csv
  - models_v2/asof_comparison/per_row_predictions_v2b.csv
  - reports/figures/asof_mape_v2_vs_v2b.png
  - reports/figures/asof_rmse_v2_vs_v2b.png
  - reports/asof_v2b_results.md
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
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

AS_OF_TAGS = ["07_01", "08_01", "09_01"]
AS_OF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}

CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id", "rotation_pair"]
NON_FEATURE = ["field_id", "year", "target_yield_t_ha", "target_source", "unit_fix_applied", "standard_name"]
MIN_TRAIN_YEARS = 4
MIN_TRAIN_ROWS = 30
WALK_FORWARD_TEST_YEARS = list(range(2017, 2026))

CATBOOST_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
    verbose=False,
)

CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
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
    return X, df["target_yield_t_ha"].astype(float), feat_cols


def walk_forward_oof(df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    X, y, feat_cols = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in CAT_FEATURES if c in feat_cols]
    years = df["year"].values
    oof = np.full(len(df), np.nan)
    folds = []
    for ty in WALK_FORWARD_TEST_YEARS:
        test_mask = years == ty
        train_mask = years < ty
        if test_mask.sum() == 0:
            continue
        if train_mask.sum() < MIN_TRAIN_ROWS or len(pd.unique(years[train_mask])) < MIN_TRAIN_YEARS:
            continue
        Xtr, ytr = X.iloc[train_mask], y.iloc[train_mask]
        Xte, yte = X.iloc[test_mask], y.iloc[test_mask]
        n = len(Xtr)
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        rng.shuffle(idx)
        v = max(5, int(n * 0.2))
        val_idx, tr_idx = idx[:v], idx[v:]
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(
            Pool(Xtr.iloc[tr_idx], ytr.iloc[tr_idx], cat_features=cat_idx),
            eval_set=Pool(Xtr.iloc[val_idx], ytr.iloc[val_idx], cat_features=cat_idx),
        )
        pred = m.predict(Xte)
        oof[test_mask] = pred
        folds.append({"test_year": int(ty), "n_train": int(n), "n_test": int(test_mask.sum())})
    return oof, folds


def metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    if len(y) < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "mape": float(np.mean(np.abs((y - p) / np.where(y > 0, y, np.nan))) * 100),
    }


def run_scenario(scenario: str, crop_filter: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'='*80}\nSCENARIO {scenario} — v2b\n{'='*80}")
    rows = []
    pred_rows = []

    for tag in AS_OF_TAGS:
        v2_path = DATA_PROCESSED / f"ml_dataset_clean_v2_asof_{tag}.csv"
        v2b_path = DATA_PROCESSED / f"ml_dataset_clean_v2b_asof_{tag}.csv"
        if not v2b_path.exists():
            print(f"  Missing {v2b_path}, skip")
            continue

        df_v2 = pd.read_csv(v2_path).merge(CROPS_DF, on="crop_id", how="left")
        df_v2b = pd.read_csv(v2b_path).merge(CROPS_DF, on="crop_id", how="left")

        if crop_filter is not None:
            df_v2 = df_v2[df_v2["standard_name"].isin(crop_filter)].copy()
            df_v2b = df_v2b[df_v2b["standard_name"].isin(crop_filter)].copy()
        df_v2 = df_v2.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        df_v2b = df_v2b.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)

        if len(df_v2b) < 50:
            print(f"  {tag}: skip ({len(df_v2b)} rows)")
            continue

        oof_v2, _ = walk_forward_oof(df_v2)
        oof_v2b, _ = walk_forward_oof(df_v2b)

        # Comparable rows: both predictions available and same (field_id, year)
        df_v2 = df_v2.assign(ml_v2=oof_v2)
        df_v2b = df_v2b.assign(ml_v2b=oof_v2b)
        keys = ["field_id", "year"]
        merged = df_v2b.merge(df_v2[keys + ["ml_v2"]], on=keys, how="inner")
        merged = merged.dropna(subset=["ml_v2", "ml_v2b"])
        n = len(merged)
        if n == 0:
            print(f"  {tag}: nothing comparable")
            continue

        m_v2 = metrics(merged["target_yield_t_ha"].values, merged["ml_v2"].values)
        m_v2b = metrics(merged["target_yield_t_ha"].values, merged["ml_v2b"].values)

        rows.append({
            "scenario": scenario, "asof_tag": tag, "asof_label": AS_OF_LABELS[tag],
            "n_compare": n,
            "v2_r2": m_v2["r2"], "v2_rmse": m_v2["rmse"], "v2_mae": m_v2["mae"], "v2_mape": m_v2["mape"],
            "v2b_r2": m_v2b["r2"], "v2b_rmse": m_v2b["rmse"], "v2b_mae": m_v2b["mae"], "v2b_mape": m_v2b["mape"],
        })

        for _, r in merged.iterrows():
            pred_rows.append({
                "scenario": scenario, "asof_tag": tag,
                "field_id": r["field_id"], "year": r["year"],
                "crop_id": r["crop_id"], "standard_name": r.get("standard_name"),
                "target_yield_t_ha": r["target_yield_t_ha"],
                "ml_v2_t_ha": r["ml_v2"], "ml_v2b_t_ha": r["ml_v2b"],
            })

        print(
            f"  {tag} ({AS_OF_LABELS[tag]}): n={n}  "
            f"v2 R²={m_v2['r2']:.3f} MAPE={m_v2['mape']:.1f}%  →  "
            f"v2b R²={m_v2b['r2']:.3f} MAPE={m_v2b['mape']:.1f}%"
        )

    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def plot_curves(summary: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    if summary.empty:
        return paths
    for metric_pair, ylabel, fname in [
        (("v2_mape", "v2b_mape"), "MAPE, %", "asof_mape_v2_vs_v2b.png"),
        (("v2_rmse", "v2b_rmse"), "RMSE, т/га", "asof_rmse_v2_vs_v2b.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for sc in summary["scenario"].unique():
            s = summary[summary["scenario"] == sc].sort_values("asof_tag")
            plt.plot(s["asof_label"], s[metric_pair[0]], marker="o", lw=2, label=f"v2 — {sc}")
            plt.plot(s["asof_label"], s[metric_pair[1]], marker="s", lw=2, linestyle="--", label=f"v2b (agronomic) — {sc}")
        plt.xlabel("As-of forecast date")
        plt.ylabel(ylabel)
        plt.title(f"Effect of agronomic features ({ylabel.split(',')[0]})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = FIGS / fname
        plt.savefig(p, dpi=140)
        plt.close()
        paths[fname] = p
    return paths


def main():
    print("=" * 80)
    print("AS-OF v2 vs v2b (agronomic features) — Sprint 3.2")
    print("=" * 80)
    summaries = []
    preds = []
    for sc, cf in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]:
        s, p = run_scenario(sc, cf)
        if not s.empty:
            summaries.append(s)
            preds.append(p)
    if not summaries:
        print("No results")
        return
    summary = pd.concat(summaries, ignore_index=True)
    pred = pd.concat(preds, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v2b.csv", index=False)
    pred.to_csv(COMP_DIR / "per_row_predictions_v2b.csv", index=False)
    paths = plot_curves(summary)

    md = ["# As-of v2 vs v2b — agronomic feature impact\n\n"]
    md.append("_Generated by `scripts/asof/train_compare_asof_v2b.py`_\n\n")
    md.append("## New features in v2b\n")
    md.append("- NDVI: integral, peak DOY, amplitude, days above 0.5\n")
    md.append("- Weather: longest drought run, longest heat run, monthly precip/hot for Jul/Aug\n")
    md.append("- Field history: years_since_sunflower, years_since_wheat, rotation_pair (categorical)\n\n")
    md.append("## Results\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    for k, p in paths.items():
        md.append(f"### {k}\n\n![{k}]({p.relative_to(REPORTS).as_posix()})\n\n")
    out_md = REPORTS / "asof_v2b_results.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL")
    print("=" * 80)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v2b.csv'}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
