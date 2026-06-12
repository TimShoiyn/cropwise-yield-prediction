"""
Sprint 5.3 — v5 (v4 + GDD-aligned NDVI) vs v4 vs Cropwise.

Outputs:
  - models_v2/asof_comparison/asof_results_v5.csv
  - models_v2/asof_comparison/per_row_predictions_v5.csv
  - reports/figures/asof_mape_v4_vs_v5.png
  - reports/figures/asof_rmse_v4_vs_v5.png
  - reports/asof_v5_results.md
"""

from __future__ import annotations

import ast
import json
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
    iterations=600, learning_rate=0.05, depth=5, l2_leaf_reg=3.0,
    random_seed=42, loss_function="RMSE", od_type="Iter", od_wait=50,
    allow_writing_files=False, verbose=False,
)

MAX_T_HA = {
    "sunflower": 5.0, "wheat_spring": 7.0, "wheat_winter": 8.0, "barley_spring": 7.0,
    "maize": 12.0, "oil_seed_raps_spring": 4.5, "oil_seed_raps_winter": 5.0,
    "soya": 4.0, "pea": 4.5, "lentil": 3.5, "buckwheat": 3.0, "safflower": 2.5,
}
DEFAULT_MAX_T_HA = 5.0
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


def walk_forward_oof(df: pd.DataFrame) -> np.ndarray:
    X, y, feat_cols = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in CAT_FEATURES if c in feat_cols]
    years = df["year"].values
    oof = np.full(len(df), np.nan)
    for ty in WALK_FORWARD_TEST_YEARS:
        test_mask = years == ty
        train_mask = years < ty
        if test_mask.sum() == 0 or train_mask.sum() < MIN_TRAIN_ROWS or len(pd.unique(years[train_mask])) < MIN_TRAIN_YEARS:
            continue
        Xtr, ytr = X.iloc[train_mask], y.iloc[train_mask]
        Xte = X.iloc[test_mask]
        n = len(Xtr)
        rng = np.random.default_rng(42)
        idx = np.arange(n); rng.shuffle(idx)
        v = max(5, int(n * 0.2))
        val_idx, tr_idx = idx[:v], idx[v:]
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(
            Pool(Xtr.iloc[tr_idx], ytr.iloc[tr_idx], cat_features=cat_idx),
            eval_set=Pool(Xtr.iloc[val_idx], ytr.iloc[val_idx], cat_features=cat_idx),
        )
        oof[test_mask] = m.predict(Xte)
    return oof


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


def parse_estimate_history(s) -> dict:
    if pd.isna(s): return {}
    if isinstance(s, dict): return s
    text = str(s)
    try: return ast.literal_eval(text)
    except Exception:
        try: return json.loads(text.replace("'", '"'))
        except Exception: return {}


def cropwise_asof_value(history: dict, as_of_date: pd.Timestamp) -> float | None:
    if not history: return None
    parsed = []
    for k, v in history.items():
        d = pd.to_datetime(k, errors="coerce")
        if pd.isna(d): continue
        try: val = float(v)
        except Exception: continue
        parsed.append((d, val))
    eligible = [(d, v) for d, v in parsed if d <= as_of_date]
    if not eligible: return None
    eligible.sort(key=lambda x: x[0])
    return eligible[-1][1]


def normalize_cropwise(value: float | None, std_name: str | None) -> float | None:
    if value is None: return None
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if value > cap: return value / 10.0
    return float(value)


def extract_cropwise_asof_table() -> pd.DataFrame:
    peh = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    peh["estimate_history_dict"] = peh["estimate_history"].apply(parse_estimate_history)
    peh = peh[["field_id", "year", "estimate_history_dict"]].copy()
    peh["field_id"] = peh["field_id"].astype(int)
    peh["year"] = peh["year"].astype(int)
    return peh


def run_scenario(scenario: str, crop_filter: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'='*80}\nSCENARIO {scenario} — v4 vs v5 vs Cropwise\n{'='*80}")
    rows, pred_rows = [], []
    cw_table = extract_cropwise_asof_table()

    for tag in AS_OF_TAGS:
        v4_path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{tag}.csv"
        v5_path = DATA_PROCESSED / f"ml_dataset_clean_v5_asof_{tag}.csv"
        if not v5_path.exists():
            print(f"  Missing {v5_path}, skip"); continue

        df_v4 = pd.read_csv(v4_path).merge(CROPS_DF, on="crop_id", how="left")
        df_v5 = pd.read_csv(v5_path).merge(CROPS_DF, on="crop_id", how="left")

        if crop_filter is not None:
            df_v4 = df_v4[df_v4["standard_name"].isin(crop_filter)].copy()
            df_v5 = df_v5[df_v5["standard_name"].isin(crop_filter)].copy()
        df_v4 = df_v4.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        df_v5 = df_v5.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)

        if len(df_v5) < 50:
            print(f"  {tag}: skip ({len(df_v5)} rows)"); continue

        oof_v4 = walk_forward_oof(df_v4)
        oof_v5 = walk_forward_oof(df_v5)

        df_v4 = df_v4.assign(ml_v4=oof_v4)
        df_v5 = df_v5.assign(ml_v5=oof_v5)

        keys = ["field_id", "year"]
        merged = df_v5.merge(df_v4[keys + ["ml_v4"]], on=keys, how="inner")
        merged = merged.merge(cw_table, on=keys, how="left")

        mm, dd = int(tag.split("_")[0]), int(tag.split("_")[1])
        cw_vals = []
        for _, r in merged.iterrows():
            hist = r.get("estimate_history_dict")
            v = cropwise_asof_value(hist if isinstance(hist, dict) else {}, pd.Timestamp(year=int(r["year"]), month=mm, day=dd))
            cw_vals.append(normalize_cropwise(v, r.get("standard_name")))
        merged["cropwise_asof_t_ha"] = cw_vals

        cmp_ml = merged.dropna(subset=["ml_v4", "ml_v5"])
        cmp_all = merged.dropna(subset=["ml_v4", "ml_v5", "cropwise_asof_t_ha"])
        if len(cmp_ml) == 0:
            print(f"  {tag}: nothing comparable"); continue

        m_v4 = metrics(cmp_ml["target_yield_t_ha"].values, cmp_ml["ml_v4"].values)
        m_v5 = metrics(cmp_ml["target_yield_t_ha"].values, cmp_ml["ml_v5"].values)
        m_v4_a = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["ml_v4"].values) if len(cmp_all) else {}
        m_v5_a = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["ml_v5"].values) if len(cmp_all) else {}
        m_cw = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["cropwise_asof_t_ha"].values) if len(cmp_all) else {}

        rows.append({
            "scenario": scenario, "asof_tag": tag, "asof_label": AS_OF_LABELS[tag],
            "n_ml": len(cmp_ml), "n_all": len(cmp_all),
            "v4_r2": m_v4["r2"], "v4_rmse": m_v4["rmse"], "v4_mae": m_v4["mae"], "v4_mape": m_v4["mape"],
            "v5_r2": m_v5["r2"], "v5_rmse": m_v5["rmse"], "v5_mae": m_v5["mae"], "v5_mape": m_v5["mape"],
            "v4_a_mape": m_v4_a.get("mape", float("nan")),
            "v5_a_mape": m_v5_a.get("mape", float("nan")),
            "cw_r2": m_cw.get("r2", float("nan")), "cw_rmse": m_cw.get("rmse", float("nan")),
            "cw_mae": m_cw.get("mae", float("nan")), "cw_mape": m_cw.get("mape", float("nan")),
        })

        for _, r in cmp_ml.iterrows():
            pred_rows.append({
                "scenario": scenario, "asof_tag": tag,
                "field_id": r["field_id"], "year": r["year"],
                "crop_id": r["crop_id"], "standard_name": r.get("standard_name"),
                "target_yield_t_ha": r["target_yield_t_ha"],
                "ml_v4_t_ha": r["ml_v4"], "ml_v5_t_ha": r["ml_v5"],
                "cropwise_asof_t_ha": r.get("cropwise_asof_t_ha"),
            })

        print(
            f"  {tag} ({AS_OF_LABELS[tag]}) n={len(cmp_ml)}  "
            f"v4 MAPE={m_v4['mape']:.1f}% R²={m_v4['r2']:.3f}  → "
            f"v5 MAPE={m_v5['mape']:.1f}% R²={m_v5['r2']:.3f}  | "
            f"CW MAPE={m_cw.get('mape', float('nan')):.1f}%"
        )

    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def plot_curves(summary: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    if summary.empty: return paths
    for cols, ylabel, fname in [
        (("v4_mape", "v5_mape", "cw_mape"), "MAPE, %", "asof_mape_v4_vs_v5.png"),
        (("v4_rmse", "v5_rmse", "cw_rmse"), "RMSE, t/ha", "asof_rmse_v4_vs_v5.png"),
    ]:
        plt.figure(figsize=(9, 5.5))
        for sc in summary["scenario"].unique():
            s = summary[summary["scenario"] == sc].sort_values("asof_tag")
            plt.plot(s["asof_label"], s[cols[0]], marker="o", lw=2, label=f"v4 — {sc}")
            plt.plot(s["asof_label"], s[cols[1]], marker="^", lw=2.5, label=f"v5 (GDD-aligned) — {sc}")
            plt.plot(s["asof_label"], s[cols[2]], marker="s", lw=2, linestyle="--", color="grey", label=f"Cropwise — {sc}")
        plt.xlabel("As-of forecast date"); plt.ylabel(ylabel)
        plt.title(f"GDD-aligned NDVI effect ({ylabel.split(',')[0]})")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        p = FIGS / fname; plt.savefig(p, dpi=140); plt.close()
        paths[fname] = p
    return paths


def main():
    print("=" * 80)
    print("AS-OF v4 vs v5 (GDD-aligned NDVI) vs Cropwise — Sprint 5.3")
    print("=" * 80)
    summaries, preds = [], []
    for sc, cf in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]:
        s, p = run_scenario(sc, cf)
        if not s.empty: summaries.append(s); preds.append(p)
    if not summaries: print("No results"); return
    summary = pd.concat(summaries, ignore_index=True)
    pred = pd.concat(preds, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v5.csv", index=False)
    pred.to_csv(COMP_DIR / "per_row_predictions_v5.csv", index=False)
    paths = plot_curves(summary)

    md = ["# Sprint 5.3 — GDD-aligned NDVI: v4 → v5 vs Cropwise\n\n"]
    md.append("_Generated by `scripts/asof/train_compare_asof_v5.py`_\n\n")
    md.append("## What changed in v5\n")
    md.append("Added phenology-aligned features instead of (or in addition to) calendar-aligned NDVI:\n\n")
    md.append("- `gdd_at_asof` — accumulated GDD (base 5°C) from April 1 to as-of date\n")
    md.append("- `ndvi_at_gdd_{200..1400}` — NDVI interpolated at fixed phenological stages\n")
    md.append("- `ndvi_peak_gdd_at_asof` — at which GDD did season-to-date peak NDVI occur\n")
    md.append("- `ndvi_peak_value_at_asof` — peak NDVI value (smoothed)\n")
    md.append("- `ndvi_growth_per_100gdd` — early-season slope (emergence → peak)\n\n")
    md.append("Same biophysical stage = same feature column across years, regardless of calendar shift.\n\n")
    md.append("## Results\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    for k, p in paths.items():
        md.append(f"### {k}\n\n![{k}]({p.relative_to(REPORTS).as_posix()})\n\n")
    out_md = REPORTS / "asof_v5_results.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print("\n" + "=" * 80); print("FINAL"); print("=" * 80)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v5.csv'}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
