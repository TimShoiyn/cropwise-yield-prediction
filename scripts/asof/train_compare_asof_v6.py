"""
Sprint 6 (early) — v6 (minimal cleaned set) vs v4 (full set) vs Cropwise.

Tests the hypothesis that removing redundant + low-importance features improves
generalization (less overfitting) on small per-crop subsets.

Outputs:
  - models_v2/asof_comparison/asof_results_v6.csv
  - reports/asof_v6_results.md
  - reports/figures/asof_mape_v4_vs_v6.png
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
    return X, df["target_yield_t_ha"].astype(float), feat_cols


def walk_forward_oof(df: pd.DataFrame) -> np.ndarray:
    X, y, feat_cols = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in CAT_FEATURES if c in feat_cols]
    years = df["year"].values
    oof = np.full(len(df), np.nan)
    for ty in WALK_FORWARD_TEST_YEARS:
        test_mask = years == ty
        train_mask = years < ty
        if test_mask.sum() == 0 or train_mask.sum() < 30 or len(pd.unique(years[train_mask])) < 4:
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


def parse_estimate_history(s):
    if pd.isna(s): return {}
    if isinstance(s, dict): return s
    text = str(s)
    try: return ast.literal_eval(text)
    except Exception:
        try: return json.loads(text.replace("'", '"'))
        except Exception: return {}


def cropwise_asof_value(history, as_of_date):
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


def normalize_cropwise(value, std_name):
    if value is None: return None
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if value > cap: return value / 10.0
    return float(value)


def extract_cropwise_asof_table():
    peh = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    peh["estimate_history_dict"] = peh["estimate_history"].apply(parse_estimate_history)
    peh = peh[["field_id", "year", "estimate_history_dict"]].copy()
    peh["field_id"] = peh["field_id"].astype(int)
    peh["year"] = peh["year"].astype(int)
    return peh


def run_scenario(scenario, crop_filter):
    print(f"\n{'='*80}\nSCENARIO {scenario} — v4 vs v6 vs Cropwise\n{'='*80}")
    rows = []
    cw_table = extract_cropwise_asof_table()
    for tag in AS_OF_TAGS:
        v4_path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{tag}.csv"
        v6_path = DATA_PROCESSED / f"ml_dataset_clean_v6_asof_{tag}.csv"
        if not v6_path.exists():
            continue
        df_v4 = pd.read_csv(v4_path).merge(CROPS_DF, on="crop_id", how="left")
        df_v6 = pd.read_csv(v6_path).merge(CROPS_DF, on="crop_id", how="left")
        if crop_filter is not None:
            df_v4 = df_v4[df_v4["standard_name"].isin(crop_filter)].copy()
            df_v6 = df_v6[df_v6["standard_name"].isin(crop_filter)].copy()
        df_v4 = df_v4.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        df_v6 = df_v6.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        if len(df_v6) < 50:
            continue
        oof_v4 = walk_forward_oof(df_v4)
        oof_v6 = walk_forward_oof(df_v6)
        df_v4 = df_v4.assign(ml_v4=oof_v4)
        df_v6 = df_v6.assign(ml_v6=oof_v6)
        merged = df_v6.merge(df_v4[["field_id", "year", "ml_v4"]], on=["field_id", "year"], how="inner")
        merged = merged.merge(cw_table, on=["field_id", "year"], how="left")
        mm, dd = int(tag.split("_")[0]), int(tag.split("_")[1])
        cw_vals = []
        for _, r in merged.iterrows():
            hist = r.get("estimate_history_dict")
            v = cropwise_asof_value(hist if isinstance(hist, dict) else {}, pd.Timestamp(year=int(r["year"]), month=mm, day=dd))
            cw_vals.append(normalize_cropwise(v, r.get("standard_name")))
        merged["cropwise_asof_t_ha"] = cw_vals
        cmp_ml = merged.dropna(subset=["ml_v4", "ml_v6"])
        cmp_all = merged.dropna(subset=["ml_v4", "ml_v6", "cropwise_asof_t_ha"])
        if len(cmp_ml) == 0: continue
        m4 = metrics(cmp_ml["target_yield_t_ha"], cmp_ml["ml_v4"])
        m6 = metrics(cmp_ml["target_yield_t_ha"], cmp_ml["ml_v6"])
        mcw = metrics(cmp_all["target_yield_t_ha"], cmp_all["cropwise_asof_t_ha"]) if len(cmp_all) else {}
        rows.append({
            "scenario": scenario, "asof_tag": tag, "asof_label": AS_OF_LABELS[tag],
            "n_ml": len(cmp_ml), "n_all": len(cmp_all),
            "v4_mape": m4["mape"], "v6_mape": m6["mape"], "cw_mape": mcw.get("mape", float("nan")),
            "v4_r2": m4["r2"], "v6_r2": m6["r2"], "cw_r2": mcw.get("r2", float("nan")),
            "v4_rmse": m4["rmse"], "v6_rmse": m6["rmse"], "cw_rmse": mcw.get("rmse", float("nan")),
        })
        print(
            f"  {tag} ({AS_OF_LABELS[tag]}) n={len(cmp_ml)}  "
            f"v4 MAPE={m4['mape']:.1f}% R²={m4['r2']:.3f}  → "
            f"v6 MAPE={m6['mape']:.1f}% R²={m6['r2']:.3f}  | "
            f"CW MAPE={mcw.get('mape', float('nan')):.1f}%"
        )
    return pd.DataFrame(rows)


def plot_curves(summary):
    if summary.empty: return
    plt.figure(figsize=(9, 5.5))
    for sc in summary["scenario"].unique():
        s = summary[summary["scenario"] == sc].sort_values("asof_tag")
        plt.plot(s["asof_label"], s["v4_mape"], marker="o", lw=2, label=f"v4 — {sc}")
        plt.plot(s["asof_label"], s["v6_mape"], marker="^", lw=2.5, label=f"v6 (clean) — {sc}")
        plt.plot(s["asof_label"], s["cw_mape"], marker="s", lw=2, ls="--", color="grey", label=f"Cropwise — {sc}")
    plt.title("Effect of feature pruning (v6) — MAPE")
    plt.xlabel("As-of forecast date"); plt.ylabel("MAPE, %")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    p = FIGS / "asof_mape_v4_vs_v6.png"
    plt.savefig(p, dpi=140); plt.close()
    return p


def main():
    print("=" * 80)
    print("AS-OF v4 vs v6 (minimal) vs Cropwise — Sprint 6")
    print("=" * 80)
    summaries = []
    for sc, cf in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("wheat_winter", ["wheat_winter"]),
        ("all_crops", None),
    ]:
        s = run_scenario(sc, cf)
        if not s.empty: summaries.append(s)
    if not summaries:
        print("No results"); return
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v6.csv", index=False)
    plot_path = plot_curves(summary)
    md = ["# Sprint 6 — Feature pruning v4 → v6\n\n"]
    md.append("Removed 15 redundant or low-signal features identified in `AUDIT.md`:\n\n")
    md.append("- duplicates (Spearman ≥ 0.95): `field_calculated_area`, `wx_srad_mean_to_asof`,\n")
    md.append("  `wx_gdd_wheat_to_asof`, `ndvi_p10_max_asof`, all `wx_*_last30d` variants\n")
    md.append("- near-zero importance: `wx_hot_d35_to_asof`, `wx_apr_*`, `wx_may_hot_d30`\n\n")
    md.append("## Results\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    if plot_path:
        md.append(f"![v4 vs v6 MAPE](figures/{plot_path.name})\n")
    (REPORTS / "asof_v6_results.md").write_text("".join(md), encoding="utf-8")
    print("\n" + "=" * 80); print("FINAL"); print("=" * 80)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v6.csv'}")
    print(f"Saved {REPORTS / 'asof_v6_results.md'}")


if __name__ == "__main__":
    main()
