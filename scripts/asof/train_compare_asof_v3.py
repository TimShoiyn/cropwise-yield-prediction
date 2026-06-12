"""
Sprint 5.1 — Train as-of CatBoost on v3 (Open-Meteo backed) and 3-way compare:
    v2b  (Sprint 3.2 features, sparse weather)
    v3   (Sprint 5.1 features, ERA5/Open-Meteo + srad/VPD/ET0)
    Cropwise (industry benchmark from estimate_history)

Outputs:
  - models_v2/asof_comparison/asof_results_v3.csv
  - models_v2/asof_comparison/per_row_predictions_v3.csv
  - reports/figures/asof_mape_v2b_vs_v3.png
  - reports/figures/asof_rmse_v2b_vs_v3.png
  - reports/asof_v3_results.md
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

MAX_T_HA = {
    "sunflower": 5.0, "wheat_spring": 7.0, "wheat_winter": 8.0, "barley_spring": 7.0,
    "maize": 12.0, "oil_seed_raps_spring": 4.5, "oil_seed_raps_winter": 5.0,
    "soya": 4.0, "pea": 4.5, "lentil": 3.5, "buckwheat": 3.0, "safflower": 2.5,
}
DEFAULT_MAX_T_HA = 5.0

CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


# ---------- ML walk-forward ----------

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


# ---------- Cropwise as-of ----------

def parse_estimate_history(s) -> dict:
    if pd.isna(s):
        return {}
    if isinstance(s, dict):
        return s
    text = str(s)
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text.replace("'", '"'))
        except Exception:
            return {}


def cropwise_asof_value(history: dict, as_of_date: pd.Timestamp) -> float | None:
    if not history:
        return None
    parsed = []
    for k, v in history.items():
        d = pd.to_datetime(k, errors="coerce")
        if pd.isna(d):
            continue
        try:
            val = float(v)
        except Exception:
            continue
        parsed.append((d, val))
    eligible = [(d, v) for d, v in parsed if d <= as_of_date]
    if not eligible:
        return None
    eligible.sort(key=lambda x: x[0])
    return eligible[-1][1]


def normalize_cropwise(value: float | None, std_name: str | None) -> float | None:
    if value is None:
        return None
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if value > cap:
        return value / 10.0
    return float(value)


def extract_cropwise_asof_table() -> pd.DataFrame:
    peh = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    peh["estimate_history_dict"] = peh["estimate_history"].apply(parse_estimate_history)
    peh = peh[["field_id", "year", "estimate_history_dict"]].copy()
    peh["field_id"] = peh["field_id"].astype(int)
    peh["year"] = peh["year"].astype(int)
    return peh


# ---------- Pipeline ----------

def run_scenario(scenario: str, crop_filter: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'='*80}\nSCENARIO {scenario} — v2b vs v3 vs Cropwise\n{'='*80}")
    rows: list[dict] = []
    pred_rows: list[dict] = []
    cw_table = extract_cropwise_asof_table()

    for tag in AS_OF_TAGS:
        v2b_path = DATA_PROCESSED / f"ml_dataset_clean_v2b_asof_{tag}.csv"
        v3_path = DATA_PROCESSED / f"ml_dataset_clean_v3_asof_{tag}.csv"
        if not v3_path.exists():
            print(f"  Missing {v3_path}, skip"); continue

        df_v2b = pd.read_csv(v2b_path).merge(CROPS_DF, on="crop_id", how="left")
        df_v3 = pd.read_csv(v3_path).merge(CROPS_DF, on="crop_id", how="left")

        if crop_filter is not None:
            df_v2b = df_v2b[df_v2b["standard_name"].isin(crop_filter)].copy()
            df_v3 = df_v3[df_v3["standard_name"].isin(crop_filter)].copy()
        df_v2b = df_v2b.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        df_v3 = df_v3.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)

        if len(df_v3) < 50:
            print(f"  {tag}: skip ({len(df_v3)} rows)"); continue

        oof_v2b, _ = walk_forward_oof(df_v2b)
        oof_v3, _ = walk_forward_oof(df_v3)

        df_v2b = df_v2b.assign(ml_v2b=oof_v2b)
        df_v3 = df_v3.assign(ml_v3=oof_v3)

        keys = ["field_id", "year"]
        merged = df_v3.merge(df_v2b[keys + ["ml_v2b"]], on=keys, how="inner")

        # join cropwise asof
        merged = merged.merge(cw_table, on=keys, how="left")
        mm, dd = int(tag.split("_")[0]), int(tag.split("_")[1])
        cw_vals = []
        for _, r in merged.iterrows():
            hist = r.get("estimate_history_dict")
            v = cropwise_asof_value(hist if isinstance(hist, dict) else {}, pd.Timestamp(year=int(r["year"]), month=mm, day=dd))
            cw_vals.append(normalize_cropwise(v, r.get("standard_name")))
        merged["cropwise_asof_t_ha"] = cw_vals

        # we want metrics on rows where ALL THREE predictions exist (apples-to-apples)
        cmp_all = merged.dropna(subset=["ml_v2b", "ml_v3", "cropwise_asof_t_ha"])
        n_all = len(cmp_all)

        # also a "ml-only" comparison (drop cropwise NaN constraint): v2b vs v3 only
        cmp_ml = merged.dropna(subset=["ml_v2b", "ml_v3"])
        n_ml = len(cmp_ml)

        if n_ml == 0:
            print(f"  {tag}: nothing comparable"); continue

        m_v2b = metrics(cmp_ml["target_yield_t_ha"].values, cmp_ml["ml_v2b"].values)
        m_v3 = metrics(cmp_ml["target_yield_t_ha"].values, cmp_ml["ml_v3"].values)

        # cropwise metrics on the ALL-THREE subset (otherwise unfair to compare on different rows)
        if n_all > 0:
            m_v2b_a = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["ml_v2b"].values)
            m_v3_a = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["ml_v3"].values)
            m_cw = metrics(cmp_all["target_yield_t_ha"].values, cmp_all["cropwise_asof_t_ha"].values)
        else:
            m_v2b_a = {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
            m_v3_a = m_v2b_a
            m_cw = m_v2b_a

        rows.append({
            "scenario": scenario, "asof_tag": tag, "asof_label": AS_OF_LABELS[tag],
            "n_ml": n_ml, "n_all": n_all,
            # ml vs ml on full ml-comparable set
            "v2b_r2": m_v2b["r2"], "v2b_rmse": m_v2b["rmse"], "v2b_mae": m_v2b["mae"], "v2b_mape": m_v2b["mape"],
            "v3_r2": m_v3["r2"], "v3_rmse": m_v3["rmse"], "v3_mae": m_v3["mae"], "v3_mape": m_v3["mape"],
            # all 3 on the cropwise-comparable subset
            "v2b_a_rmse": m_v2b_a["rmse"], "v2b_a_mape": m_v2b_a["mape"],
            "v3_a_rmse": m_v3_a["rmse"], "v3_a_mape": m_v3_a["mape"],
            "cw_r2": m_cw["r2"], "cw_rmse": m_cw["rmse"], "cw_mae": m_cw["mae"], "cw_mape": m_cw["mape"],
        })

        for _, r in cmp_ml.iterrows():
            pred_rows.append({
                "scenario": scenario, "asof_tag": tag,
                "field_id": r["field_id"], "year": r["year"],
                "crop_id": r["crop_id"], "standard_name": r.get("standard_name"),
                "target_yield_t_ha": r["target_yield_t_ha"],
                "ml_v2b_t_ha": r["ml_v2b"], "ml_v3_t_ha": r["ml_v3"],
                "cropwise_asof_t_ha": r.get("cropwise_asof_t_ha"),
            })

        print(
            f"  {tag} ({AS_OF_LABELS[tag]}) n_ml={n_ml} n_all={n_all}  "
            f"v2b MAPE={m_v2b['mape']:.1f}%  → v3 MAPE={m_v3['mape']:.1f}%  | "
            f"CW MAPE={m_cw['mape']:.1f}%"
        )

    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def plot_curves(summary: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    if summary.empty:
        return paths
    for cols, ylabel, fname in [
        (("v2b_mape", "v3_mape", "cw_mape"), "MAPE, %", "asof_mape_v2b_vs_v3.png"),
        (("v2b_rmse", "v3_rmse", "cw_rmse"), "RMSE, t/ha", "asof_rmse_v2b_vs_v3.png"),
    ]:
        plt.figure(figsize=(9, 5.5))
        for sc in summary["scenario"].unique():
            s = summary[summary["scenario"] == sc].sort_values("asof_tag")
            plt.plot(s["asof_label"], s[cols[0]], marker="o", lw=2, label=f"v2b — {sc}")
            plt.plot(s["asof_label"], s[cols[1]], marker="^", lw=2.5, label=f"v3 (Open-Meteo) — {sc}")
            plt.plot(s["asof_label"], s[cols[2]], marker="s", lw=2, linestyle="--", color="grey", label=f"Cropwise — {sc}")
        plt.xlabel("As-of forecast date")
        plt.ylabel(ylabel)
        plt.title(f"Open-Meteo backfill effect ({ylabel.split(',')[0]})")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = FIGS / fname
        plt.savefig(p, dpi=140)
        plt.close()
        paths[fname] = p
    return paths


def main():
    print("=" * 80)
    print("AS-OF v2b vs v3 (Open-Meteo) vs Cropwise — Sprint 5.1")
    print("=" * 80)
    summaries: list[pd.DataFrame] = []
    preds: list[pd.DataFrame] = []
    for sc, cf in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]:
        s, p = run_scenario(sc, cf)
        if not s.empty:
            summaries.append(s); preds.append(p)
    if not summaries:
        print("No results"); return
    summary = pd.concat(summaries, ignore_index=True)
    pred = pd.concat(preds, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v3.csv", index=False)
    pred.to_csv(COMP_DIR / "per_row_predictions_v3.csv", index=False)
    paths = plot_curves(summary)

    md = ["# Sprint 5.1 — Open-Meteo backfill: v2b → v3 vs Cropwise\n\n"]
    md.append("_Generated by `scripts/asof/train_compare_asof_v3.py`_\n\n")
    md.append("## What changed in v3\n")
    md.append("- All `wx_*` features rebuilt from **Open-Meteo / ERA5** archive (covers 2010-2025 fully).\n")
    md.append("- New physical signals not present before:\n")
    md.append("  - `wx_srad_sum_to_asof`, monthly `wx_*_srad` — solar radiation (ceiling for photosynthesis).\n")
    md.append("  - `wx_vpd_max_to_asof`, `wx_vpd_days_high_to_asof` — atmospheric drought stress.\n")
    md.append("  - `wx_et0_sum_to_asof`, `wx_water_balance_to_asof = precip − et0` — water demand vs supply.\n")
    md.append("  - GDD with crop-specific bases: sunflower (T_base=6°C), wheat (T_base=4°C).\n")
    md.append("  - Last-30-days windows for srad/precip/et0/vpd.\n\n")
    md.append("## Results — ML v2b vs ML v3 vs Cropwise\n\n")
    md.append("`v2b_*` and `v3_*` columns: metrics on the v2b∩v3 set. `cw_*`: metrics on the v2b∩v3∩Cropwise set.\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    for k, p in paths.items():
        md.append(f"### {k}\n\n![{k}]({p.relative_to(REPORTS).as_posix()})\n\n")
    out_md = REPORTS / "asof_v3_results.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL")
    print("=" * 80)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v3.csv'}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
