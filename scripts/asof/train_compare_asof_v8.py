"""
Sprint 6.1 — Honest v8 = v7 + SoilGrids vs v7 vs Cropwise.

Tests whether SoilGrids 250m soil features add to the factual-target model.
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
MIN_TRAIN_YEARS = 3
MIN_TRAIN_ROWS = 20

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


def prepare_xy(df):
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


def walk_forward_oof(df):
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


def cropwise_table():
    peh = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    peh["estimate_history_dict"] = peh["estimate_history"].apply(parse_estimate_history)
    peh = peh[["field_id", "year", "estimate_history_dict"]].copy()
    peh["field_id"] = peh["field_id"].astype(int)
    peh["year"] = peh["year"].astype(int)
    return peh


def feat_imp_top(df: pd.DataFrame, n: int = 12):
    X, y, fcols = prepare_xy(df)
    cat_idx = [fcols.index(c) for c in CAT_FEATURES if c in fcols]
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5, random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(Pool(X, y, cat_features=cat_idx))
    imp = pd.DataFrame({"feature": fcols, "importance": m.get_feature_importance()})
    return imp.sort_values("importance", ascending=False).head(n)


def run_scenario(scenario, crop_filter):
    print(f"\n{'='*80}\nSCENARIO {scenario} — v7 vs v8 (with SoilGrids) vs Cropwise\n{'='*80}")
    rows = []
    cw_table = cropwise_table()
    for tag in AS_OF_TAGS:
        v7p = DATA_PROCESSED / f"ml_dataset_clean_v7_asof_{tag}.csv"
        v8p = DATA_PROCESSED / f"ml_dataset_clean_v8_asof_{tag}.csv"
        if not v8p.exists(): continue
        d7 = pd.read_csv(v7p).merge(CROPS_DF, on="crop_id", how="left")
        d8 = pd.read_csv(v8p).merge(CROPS_DF, on="crop_id", how="left")
        if crop_filter is not None:
            d7 = d7[d7["standard_name"].isin(crop_filter)].copy()
            d8 = d8[d8["standard_name"].isin(crop_filter)].copy()
        d7 = d7.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        d8 = d8.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        if len(d8) < 30: continue
        oof7 = walk_forward_oof(d7)
        oof8 = walk_forward_oof(d8)
        d7 = d7.assign(ml_v7=oof7); d8 = d8.assign(ml_v8=oof8)
        merged = d8.merge(d7[["field_id","year","ml_v7"]], on=["field_id","year"], how="inner")
        merged = merged.merge(cw_table, on=["field_id","year"], how="left")
        mm, dd = int(tag.split("_")[0]), int(tag.split("_")[1])
        cw_vals = []
        for _, r in merged.iterrows():
            hist = r.get("estimate_history_dict")
            v = cropwise_asof_value(hist if isinstance(hist, dict) else {}, pd.Timestamp(year=int(r["year"]), month=mm, day=dd))
            cw_vals.append(normalize_cropwise(v, r.get("standard_name")))
        merged["cropwise_asof"] = cw_vals
        cmp = merged.dropna(subset=["ml_v7","ml_v8"])
        cmp_all = cmp.dropna(subset=["cropwise_asof"])
        if len(cmp) == 0: continue
        m7 = metrics(cmp["target_yield_t_ha"], cmp["ml_v7"])
        m8 = metrics(cmp["target_yield_t_ha"], cmp["ml_v8"])
        mcw = metrics(cmp_all["target_yield_t_ha"], cmp_all["cropwise_asof"]) if len(cmp_all) else {}
        rows.append({
            "scenario": scenario, "asof_tag": tag, "asof_label": AS_OF_LABELS[tag], "n": len(cmp),
            "v7_mape": m7["mape"], "v8_mape": m8["mape"], "cw_mape": mcw.get("mape", float("nan")),
            "v7_r2": m7["r2"], "v8_r2": m8["r2"], "cw_r2": mcw.get("r2", float("nan")),
            "v7_rmse": m7["rmse"], "v8_rmse": m8["rmse"], "cw_rmse": mcw.get("rmse", float("nan")),
        })
        print(
            f"  {tag} ({AS_OF_LABELS[tag]}) n={len(cmp)}  "
            f"v7 MAPE={m7['mape']:.1f}% → v8 MAPE={m8['mape']:.1f}%  | "
            f"CW MAPE={mcw.get('mape', float('nan')):.1f}%"
        )
    return pd.DataFrame(rows)


def plot_curves(summary):
    if summary.empty: return None
    plt.figure(figsize=(9, 5.5))
    for sc in summary["scenario"].unique():
        s = summary[summary["scenario"] == sc].sort_values("asof_tag")
        plt.plot(s["asof_label"], s["v7_mape"], marker="o", lw=2, label=f"v7 — {sc}")
        plt.plot(s["asof_label"], s["v8_mape"], marker="^", lw=2.5, label=f"v8 (SoilGrids) — {sc}")
        plt.plot(s["asof_label"], s["cw_mape"], marker="s", lw=2, ls="--", color="grey", label=f"Cropwise — {sc}")
    plt.title("v8 = v7 + SoilGrids — MAPE")
    plt.xlabel("As-of forecast date"); plt.ylabel("MAPE, %")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    p = FIGS / "asof_mape_v7_vs_v8.png"
    plt.savefig(p, dpi=140); plt.close()
    return p


def main():
    print("=" * 80)
    print("AS-OF v7 vs v8 (SoilGrids) vs Cropwise — Sprint 6.1")
    print("=" * 80)
    summaries = []
    sg_imp_print = False
    for sc, cf in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]:
        s = run_scenario(sc, cf)
        if not s.empty: summaries.append(s)
    if not summaries:
        print("No results"); return
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v8.csv", index=False)
    plot_path = plot_curves(summary)

    # Feature importance for one big-data scenario to see if soilgrids actually got picked
    print("\n--- Feature importance (all_crops, 1 Aug, v8) ---")
    df = pd.read_csv(DATA_PROCESSED / "ml_dataset_clean_v8_asof_08_01.csv").merge(CROPS_DF, on="crop_id", how="left")
    df = df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
    imp = feat_imp_top(df, n=15)
    print(imp.round(3).to_string(index=False))
    sg_used = imp[imp["feature"].str.startswith("sg_")]
    print(f"\nSoilGrids in top-15: {len(sg_used)} of {(imp['feature'].str.startswith('sg_')).sum()}")

    md = ["# Sprint 6.1 — v8 = v7 + SoilGrids 250m\n\n"]
    md.append("Adds 24 SoilGrids features (clay/sand/silt/SOC/N/bdod/CEC/pH × 3 depth layers) to v7.\n\n")
    md.append("## Results\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n## Top 15 features (all_crops, 1 Aug)\n\n")
    md.append(imp.round(3).to_markdown(index=False))
    md.append("\n\n")
    if plot_path:
        md.append(f"![v7 vs v8](figures/{plot_path.name})\n")
    (REPORTS / "asof_v8_results.md").write_text("".join(md), encoding="utf-8")
    print("\n" + "=" * 80); print("FINAL"); print("=" * 80)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v8.csv'}")


if __name__ == "__main__":
    main()
