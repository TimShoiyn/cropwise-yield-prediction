"""
Sprint 3.1 — Train as-of CatBoost models AND benchmark vs Cropwise own forecast.

For each as_of date D in {07-01, 08-01, 09-01}:
  1. Train CatBoost on the corresponding `ml_dataset_clean_v2_asof_{tag}.csv`
     using Walk-Forward CV (same protocol as Sprint 2). One model per crop scenario.
  2. Extract Cropwise's own as-of forecast from
     `data_raw/productivity_estimate_histories.csv` for the same (field_id, year):
       - parse the dict in `estimate_history`
       - pick the forecast at the latest date <= D in that year
  3. Normalize Cropwise forecast to t/ha using the same per-crop unit fix from Sprint 1.
  4. Compute MAPE / RMSE / R² for ML vs Cropwise on the same (field_id, year) test set.

Outputs:
  - models_v2/asof_comparison/asof_results.csv    (ml vs cropwise per as_of × crop)
  - models_v2/asof_comparison/per_row_predictions.csv   (full breakdown)
  - reports/figures/asof_mape_curve.png
  - reports/figures/asof_rmse_curve.png
  - reports/asof_forecast_results.md
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

CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id"]
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

# Same per-crop normalization caps as in build_clean_targets.py
MAX_T_HA = {
    "sunflower": 5.0, "wheat_spring": 7.0, "wheat_winter": 8.0, "barley_spring": 7.0,
    "maize": 12.0, "oil_seed_raps_spring": 4.5, "oil_seed_raps_winter": 5.0,
    "soya": 4.0, "pea": 4.5, "lentil": 3.5, "buckwheat": 3.0, "safflower": 2.5,
}
DEFAULT_MAX_T_HA = 5.0

CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


# ============================================================
# 1) ML training (walk-forward) — copy of Sprint 2 logic, adapted
# ============================================================

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    feat_cols = list(dict.fromkeys(feat_cols + ["field_id"]))
    X = df[feat_cols].copy()
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].apply(lambda v: "missing" if pd.isna(v) else str(int(float(v))))
    for c in [c for c in feat_cols if c not in CAT_FEATURES]:
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


# ============================================================
# 2) Cropwise as-of extraction
# ============================================================

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


def cropwise_asof_value(history: dict, as_of_date: pd.Timestamp) -> tuple[float | None, str | None]:
    """Return the latest forecast value <= as_of_date, and its date string."""
    if not history:
        return None, None
    parsed = []
    for k, v in history.items():
        try:
            d = pd.to_datetime(k, errors="coerce")
        except Exception:
            continue
        if pd.isna(d):
            continue
        try:
            val = float(v)
        except Exception:
            continue
        parsed.append((d, val))
    if not parsed:
        return None, None
    eligible = [(d, v) for d, v in parsed if d <= as_of_date]
    if not eligible:
        return None, None
    eligible.sort(key=lambda x: x[0])
    last_d, last_v = eligible[-1]
    return last_v, last_d.strftime("%Y-%m-%d")


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


# ============================================================
# 3) Pipeline
# ============================================================

def run_scenario(scenario_name: str, crop_filter: list[str] | None) -> pd.DataFrame:
    print(f"\n{'='*80}\nSCENARIO: {scenario_name}\n{'='*80}")
    cropwise_table = extract_cropwise_asof_table()

    rows: list[dict] = []
    pred_rows: list[dict] = []  # for per-row export

    for tag in AS_OF_TAGS:
        ds_path = DATA_PROCESSED / f"ml_dataset_clean_v2_asof_{tag}.csv"
        df = pd.read_csv(ds_path)
        df = df.merge(CROPS_DF, on="crop_id", how="left")
        if crop_filter is not None:
            df = df[df["standard_name"].isin(crop_filter)].copy()
        df = df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
        if len(df) < 50:
            print(f"  {tag}: skip ({len(df)} rows)")
            continue

        oof, folds = walk_forward_oof(df)
        df["ml_pred_t_ha"] = oof

        # Cropwise as-of join
        mm, dd = int(tag.split("_")[0]), int(tag.split("_")[1])
        df["cropwise_asof_t_ha"] = np.nan
        df["cropwise_asof_date"] = None
        merged = df.merge(cropwise_table, on=["field_id", "year"], how="left")

        cw_vals, cw_dates = [], []
        for _, r in merged.iterrows():
            hist = r.get("estimate_history_dict")
            if isinstance(hist, dict) and hist:
                as_of_ts = pd.Timestamp(year=int(r["year"]), month=mm, day=dd)
                v, d = cropwise_asof_value(hist, as_of_ts)
                v_norm = normalize_cropwise(v, r.get("standard_name"))
                cw_vals.append(v_norm)
                cw_dates.append(d)
            else:
                cw_vals.append(None)
                cw_dates.append(None)
        df["cropwise_asof_t_ha"] = cw_vals
        df["cropwise_asof_date"] = cw_dates
        df["asof_tag"] = tag

        # Build comparison only on rows where BOTH ml_pred and cropwise are available
        mask = df["ml_pred_t_ha"].notna() & df["cropwise_asof_t_ha"].notna()
        df_cmp = df[mask].copy()
        n_compare = len(df_cmp)
        if n_compare == 0:
            print(f"  {tag}: 0 comparable rows (ml or cropwise missing)")
            rows.append({"scenario": scenario_name, "asof_tag": tag, "n_compare": 0})
            continue

        yt = df_cmp["target_yield_t_ha"].values
        ml = df_cmp["ml_pred_t_ha"].values
        cw = df_cmp["cropwise_asof_t_ha"].values

        def metrics(y, p):
            return {
                "r2": float(r2_score(y, p)) if len(y) >= 2 else float("nan"),
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "mae": float(mean_absolute_error(y, p)),
                "mape": float(
                    np.mean(np.abs((y - p) / np.where(y > 0, y, np.nan))) * 100
                ),
            }

        ml_m = metrics(yt, ml)
        cw_m = metrics(yt, cw)

        rows.append(
            {
                "scenario": scenario_name,
                "asof_tag": tag,
                "asof_label": AS_OF_LABELS[tag],
                "n_compare": int(n_compare),
                "ml_r2": ml_m["r2"], "ml_rmse": ml_m["rmse"], "ml_mae": ml_m["mae"], "ml_mape": ml_m["mape"],
                "cw_r2": cw_m["r2"], "cw_rmse": cw_m["rmse"], "cw_mae": cw_m["mae"], "cw_mape": cw_m["mape"],
            }
        )

        for _, r in df_cmp.iterrows():
            pred_rows.append({
                "scenario": scenario_name,
                "asof_tag": tag,
                "field_id": r["field_id"],
                "year": r["year"],
                "crop_id": r["crop_id"],
                "standard_name": r.get("standard_name"),
                "target_yield_t_ha": r["target_yield_t_ha"],
                "ml_pred_t_ha": r["ml_pred_t_ha"],
                "cropwise_asof_t_ha": r["cropwise_asof_t_ha"],
                "cropwise_asof_date": r["cropwise_asof_date"],
            })

        print(
            f"  {tag} ({AS_OF_LABELS[tag]}): n={n_compare}  "
            f"ML R²={ml_m['r2']:.3f} MAPE={ml_m['mape']:.1f}%  |  "
            f"CW R²={cw_m['r2']:.3f} MAPE={cw_m['mape']:.1f}%"
        )

    summary_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    return summary_df, pred_df


def plot_curves(summary_df: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    if summary_df.empty:
        return paths

    summary_df = summary_df.sort_values(["scenario", "asof_tag"])
    scenarios = summary_df["scenario"].unique().tolist()

    # MAPE curve
    plt.figure(figsize=(8, 5))
    for sc in scenarios:
        s = summary_df[summary_df["scenario"] == sc]
        plt.plot(s["asof_label"], s["ml_mape"], marker="o", lw=2, label=f"ML — {sc}")
        plt.plot(
            s["asof_label"], s["cw_mape"], marker="s", lw=2, linestyle="--",
            label=f"Cropwise — {sc}",
        )
    plt.xlabel("As-of forecast date")
    plt.ylabel("MAPE, %")
    plt.title("ML vs Cropwise — MAPE by forecast date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = FIGS / "asof_mape_curve.png"
    plt.savefig(p, dpi=140)
    plt.close()
    paths["MAPE curve"] = p

    # RMSE curve
    plt.figure(figsize=(8, 5))
    for sc in scenarios:
        s = summary_df[summary_df["scenario"] == sc]
        plt.plot(s["asof_label"], s["ml_rmse"], marker="o", lw=2, label=f"ML — {sc}")
        plt.plot(
            s["asof_label"], s["cw_rmse"], marker="s", lw=2, linestyle="--",
            label=f"Cropwise — {sc}",
        )
    plt.xlabel("As-of forecast date")
    plt.ylabel("RMSE, т/га")
    plt.title("ML vs Cropwise — RMSE by forecast date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = FIGS / "asof_rmse_curve.png"
    plt.savefig(p, dpi=140)
    plt.close()
    paths["RMSE curve"] = p

    return paths


def write_md(summary_df: pd.DataFrame, paths: dict[str, Path]) -> Path:
    md = []
    md.append("# As-of forecast results — ML vs Cropwise\n\n")
    md.append("_Generated by `scripts/asof/train_and_compare_asof.py`_\n\n")
    md.append("## Setup\n\n")
    md.append("- For each forecast date D in {1 July, 1 Aug, 1 Sep}:\n")
    md.append("  - rebuild NDVI/weather aggregates using only data **on or before D** of that year\n")
    md.append("  - train CatBoost (cat_features=`crop_id, prev_crop_id, field_id`) with walk-forward CV by year (test years 2017..2025)\n")
    md.append("- Cropwise own forecast extracted from `productivity_estimate_histories.csv` (latest entry on or before D)\n")
    md.append("- Compare on the **same set of (field_id, year)** rows where both forecasts exist\n\n")

    md.append("## Headline\n\n")
    if summary_df.empty:
        md.append("_No comparable rows._\n")
    else:
        md.append(summary_df.round(3).to_markdown(index=False))
        md.append("\n\n")

    md.append("## Plots\n\n")
    for k, p in paths.items():
        md.append(f"### {k}\n\n![{k}]({p.relative_to(REPORTS).as_posix()})\n\n")

    md.append("## Files\n\n")
    md.append("- `models_v2/asof_comparison/asof_results.csv` — aggregated metrics\n")
    md.append("- `models_v2/asof_comparison/per_row_predictions.csv` — every comparable (field, year) row with ML & Cropwise forecasts\n")

    out = REPORTS / "asof_forecast_results.md"
    out.write_text("".join(md), encoding="utf-8")
    return out


def main():
    print("=" * 80)
    print("AS-OF FORECAST: ML vs CROPWISE — Sprint 3.1")
    print("=" * 80)

    sc_summaries = []
    sc_preds = []
    for sc_name, sc_filter in [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]:
        s, p = run_scenario(sc_name, sc_filter)
        if not s.empty:
            sc_summaries.append(s)
            sc_preds.append(p)

    if not sc_summaries:
        print("No results.")
        return
    summary_df = pd.concat(sc_summaries, ignore_index=True)
    pred_df = pd.concat(sc_preds, ignore_index=True)

    summary_df.to_csv(COMP_DIR / "asof_results.csv", index=False)
    pred_df.to_csv(COMP_DIR / "per_row_predictions.csv", index=False)
    print(f"\nSaved {COMP_DIR / 'asof_results.csv'}")
    print(f"Saved {COMP_DIR / 'per_row_predictions.csv'}")

    paths = plot_curves(summary_df)
    md_path = write_md(summary_df, paths)
    print(f"Saved {md_path}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(summary_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
