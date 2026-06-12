"""
Phase-based (GDD) features experiment.

Builds phase-based features from:
  - data_raw/weather_history_items.csv (daily weather by field_group_id)
  - data_raw/ndvi_timeseries.csv (NDVI by field_id)
and merges them into:
  - data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv

Saves:
  - data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv

Then trains 2 models with identical validation:
  - baseline: original cropwindow dataset (no phase features)
  - phases: dataset with phase-based features

Uses GroupKFold by year and saves:
  - models/models_phases_comparison.csv
  - models/oof_phases_{baseline|phases}_{allcrops|wheat|sunflower}.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Reuse proven training utilities from train_baseline_model.py (no side effects on import)
from train_baseline_model import (
    BEST_REG_GBDT_PARAMS,
    STANDARD_NAME_SUNFLOWER,
    STANDARD_NAME_WHEAT_SPRING,
    build_pipeline_with_params,
    evaluate_with_oof,
    load_dataset,
    prepare_features,
)


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_RAW_DIR = Path("data_raw")
DATA_PROCESSED_DIR = Path("data_processed")
MODELS_DIR = Path("models")

BASE_DATASET_PATH = DATA_PROCESSED_DIR / "ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv"
PHASES_DATASET_PATH = DATA_PROCESSED_DIR / "ml_dataset_full_extended_cropwindow_v1_phases.csv"

FIELDS_CSV = DATA_RAW_DIR / "fields.csv"
CROPS_CSV = DATA_RAW_DIR / "crops.csv"
NDVI_CSV = DATA_RAW_DIR / "ndvi_timeseries.csv"
WEATHER_DAILY_CSV = DATA_RAW_DIR / "weather_history_items.csv"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _standard_name_by_crop_id() -> dict[int, str]:
    if not CROPS_CSV.exists():
        return {}
    crops = pd.read_csv(CROPS_CSV)
    if "id" not in crops.columns or "standard_name" not in crops.columns:
        return {}
    tmp = crops[["id", "standard_name"]].dropna(subset=["id"]).copy()
    out: dict[int, str] = {}
    for _, r in tmp.iterrows():
        try:
            cid = int(float(r["id"]))
        except Exception:
            continue
        std = str(r.get("standard_name") or "").strip()
        if std:
            out[cid] = std
    return out


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _phase_from_cum_gdd(gdd_cum: pd.Series, thresholds: tuple[float, float, float]) -> pd.Series:
    t1, t2, t3 = thresholds
    x = pd.to_numeric(gdd_cum, errors="coerce")
    out = pd.Series(index=x.index, dtype="object")
    out.loc[x < t1] = "p1"
    out.loc[(x >= t1) & (x < t2)] = "p2"
    out.loc[x >= t2] = "p3"
    out.loc[~np.isfinite(x)] = np.nan
    return out


def build_phase_dataset() -> Path:
    print("=" * 80)
    print("BUILD PHASE FEATURES DATASET")
    print("=" * 80)

    base = _read_required_csv(BASE_DATASET_PATH)
    print(f"Base dataset: {BASE_DATASET_PATH} rows={len(base)} cols={len(base.columns)}")

    required_cols = {"field_id", "year", "crop_id", "target_yield_t_ha"}
    missing = required_cols - set(base.columns)
    if missing:
        raise RuntimeError(f"Base dataset missing required columns: {sorted(missing)}")

    fields = _read_required_csv(FIELDS_CSV)
    if "id" not in fields.columns or "field_group_id" not in fields.columns:
        raise RuntimeError("fields.csv must contain columns: id, field_group_id")

    fields_map = fields[["id", "field_group_id"]].copy().rename(columns={"id": "field_id"})
    fields_map["field_id"] = pd.to_numeric(fields_map["field_id"], errors="coerce")
    fields_map["field_group_id"] = pd.to_numeric(fields_map["field_group_id"], errors="coerce")
    fields_map = fields_map.dropna(subset=["field_id", "field_group_id"]).copy()

    base2 = base.merge(fields_map, on="field_id", how="left", validate="m:1")

    crop_id_to_std = _standard_name_by_crop_id()
    base2["standard_name"] = base2["crop_id"].map(
        lambda x: crop_id_to_std.get(int(float(x))) if pd.notna(x) else None
    )
    base2["is_wheat"] = base2["standard_name"].astype(str).str.strip().eq(STANDARD_NAME_WHEAT_SPRING)
    base2["is_sunflower"] = base2["standard_name"].astype(str).str.strip().eq(STANDARD_NAME_SUNFLOWER)

    # -----------------------------
    # Weather daily -> GDD cum + phases
    # -----------------------------
    wx = _read_required_csv(WEATHER_DAILY_CSV)
    wx_required = {"date", "temperature_avg", "temperature_max", "precipitation", "field_group_id", "year"}
    wx_missing = wx_required - set(wx.columns)
    if wx_missing:
        raise RuntimeError(f"weather_history_items.csv missing columns: {sorted(wx_missing)}")

    wx = wx[list(wx_required)].copy()
    wx["date"] = _ensure_datetime(wx, "date")
    wx["field_group_id"] = pd.to_numeric(wx["field_group_id"], errors="coerce")
    wx["year"] = pd.to_numeric(wx["year"], errors="coerce").astype("Int64")
    for c in ["temperature_avg", "temperature_max", "precipitation"]:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx = wx.dropna(subset=["date", "field_group_id", "year"]).copy()
    wx = wx.sort_values(["field_group_id", "year", "date"])

    tavg = wx["temperature_avg"]
    wx["gdd5_day"] = np.maximum(tavg - 5.0, 0.0)
    wx["gdd10_day"] = np.maximum(tavg - 10.0, 0.0)
    wx["gdd5_cum"] = wx.groupby(["field_group_id", "year"])["gdd5_day"].cumsum()
    wx["gdd10_cum"] = wx.groupby(["field_group_id", "year"])["gdd10_day"].cumsum()

    wx["phase_wheat"] = _phase_from_cum_gdd(wx["gdd5_cum"], (250.0, 600.0, 950.0))
    wx["phase_sunflower"] = _phase_from_cum_gdd(wx["gdd10_cum"], (350.0, 850.0, 1300.0))

    wx["is_hot_day"] = (wx["temperature_max"] > 30.0).astype(float)

    wheat_prec = (
        wx.dropna(subset=["phase_wheat"])
        .groupby(["field_group_id", "year", "phase_wheat"], as_index=False)["precipitation"]
        .sum()
        .rename(columns={"precipitation": "wx_precip_sum"})
    )
    wheat_prec_wide = (
        wheat_prec.pivot_table(index=["field_group_id", "year"], columns="phase_wheat", values="wx_precip_sum", aggfunc="first")
        .reset_index()
        .rename(columns={
            "p1": "wx_wheat_p1_precip_sum",
            "p2": "wx_wheat_p2_precip_sum",
            "p3": "wx_wheat_p3_precip_sum",
        })
    )

    sun_hot = (
        wx.dropna(subset=["phase_sunflower"])
        .groupby(["field_group_id", "year", "phase_sunflower"], as_index=False)["is_hot_day"]
        .sum()
        .rename(columns={"is_hot_day": "wx_hot_days"})
    )
    sun_hot_wide = (
        sun_hot.pivot_table(index=["field_group_id", "year"], columns="phase_sunflower", values="wx_hot_days", aggfunc="first")
        .reset_index()
        .rename(columns={
            "p1": "wx_sunflower_p1_hot_days",
            "p2": "wx_sunflower_p2_hot_days",
            "p3": "wx_sunflower_p3_hot_days",
        })
    )

    wx_phase_wide = wheat_prec_wide.merge(sun_hot_wide, on=["field_group_id", "year"], how="outer")

    # -----------------------------
    # NDVI -> tag phases by aligning to nearest previous weather day
    # (merge_asof can be finicky with multi-key sorting; we use a robust per-group searchsorted join)
    # -----------------------------
    ndvi = _read_required_csv(NDVI_CSV)
    ndvi_required = {"field_id", "year", "date", "ndvi_mean"}
    ndvi_missing = ndvi_required - set(ndvi.columns)
    if ndvi_missing:
        raise RuntimeError(f"ndvi_timeseries.csv missing columns: {sorted(ndvi_missing)}")

    ndvi = ndvi[list(ndvi_required)].copy()
    ndvi["date"] = _ensure_datetime(ndvi, "date")
    ndvi["field_id"] = pd.to_numeric(ndvi["field_id"], errors="coerce")
    ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce").astype("Int64")
    ndvi["ndvi_mean"] = pd.to_numeric(ndvi["ndvi_mean"], errors="coerce")
    ndvi = ndvi.dropna(subset=["field_id", "year", "date", "ndvi_mean"]).copy()

    # Filter NDVI to only (field_id,year) pairs present in base dataset to keep the job fast and aligned.
    base_field_ids = pd.to_numeric(base2["field_id"], errors="coerce").dropna().unique()
    base_years = pd.to_numeric(base2["year"], errors="coerce").dropna().unique()
    ndvi = ndvi[ndvi["field_id"].isin(base_field_ids) & ndvi["year"].isin(base_years)].copy()

    ndvi = ndvi.merge(fields_map, on="field_id", how="left", validate="m:1")
    ndvi = ndvi.dropna(subset=["field_group_id"]).copy()
    # merge_asof is strict: by-keys must have identical dtypes and be fully sorted.
    # Use plain int64 (not pandas nullable Int64) to satisfy merge_asof internals.
    ndvi["field_group_id"] = pd.to_numeric(ndvi["field_group_id"], errors="coerce")
    ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce")
    ndvi = ndvi.dropna(subset=["field_group_id", "year"]).copy()
    ndvi["field_group_id"] = ndvi["field_group_id"].astype(np.int64)
    ndvi["year"] = ndvi["year"].astype(np.int64)

    wx_key = wx[["field_group_id", "year", "date", "phase_wheat", "phase_sunflower"]].copy()
    wx_key["field_group_id"] = pd.to_numeric(wx_key["field_group_id"], errors="coerce")
    wx_key["year"] = pd.to_numeric(wx_key["year"], errors="coerce")
    wx_key["date"] = pd.to_datetime(wx_key["date"], errors="coerce")
    wx_key = wx_key.dropna(subset=["field_group_id", "year", "date"]).copy()
    wx_key["field_group_id"] = wx_key["field_group_id"].astype(np.int64)
    wx_key["year"] = wx_key["year"].astype(np.int64)

    # Filter weather to only groups/years we actually need
    needed_fg = pd.to_numeric(base2["field_group_id"], errors="coerce").dropna().unique()
    wx_key = wx_key[wx_key["field_group_id"].isin(needed_fg) & wx_key["year"].isin(base_years)].copy()

    ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
    ndvi = ndvi.dropna(subset=["date"]).copy()

    def _tag_one_group(g_ndvi: pd.DataFrame, g_wx: pd.DataFrame) -> pd.DataFrame:
        g_ndvi = g_ndvi.sort_values("date").copy()
        g_wx = g_wx.sort_values("date").copy()
        wx_dates = g_wx["date"].to_numpy(dtype="datetime64[ns]")
        nd_dates = g_ndvi["date"].to_numpy(dtype="datetime64[ns]")
        # idx = rightmost weather date <= ndvi date
        idx = np.searchsorted(wx_dates, nd_dates, side="right") - 1
        phase_w = np.full(shape=(len(g_ndvi),), fill_value=np.nan, dtype=object)
        phase_s = np.full(shape=(len(g_ndvi),), fill_value=np.nan, dtype=object)
        valid = idx >= 0
        if valid.any():
            phase_w[valid] = g_wx["phase_wheat"].to_numpy(dtype=object)[idx[valid]]
            phase_s[valid] = g_wx["phase_sunflower"].to_numpy(dtype=object)[idx[valid]]
        g_ndvi["phase_wheat"] = phase_w
        g_ndvi["phase_sunflower"] = phase_s
        return g_ndvi

    ndvi_tagged_parts: list[pd.DataFrame] = []
    for (fg, yr), g_ndvi in ndvi.groupby(["field_group_id", "year"], sort=False):
        g_wx = wx_key[(wx_key["field_group_id"] == fg) & (wx_key["year"] == yr)]
        if g_wx.empty:
            g_ndvi = g_ndvi.copy()
            g_ndvi["phase_wheat"] = np.nan
            g_ndvi["phase_sunflower"] = np.nan
            ndvi_tagged_parts.append(g_ndvi)
        else:
            ndvi_tagged_parts.append(_tag_one_group(g_ndvi, g_wx))

    ndvi_tagged = pd.concat(ndvi_tagged_parts, ignore_index=True)

    ndvi_wheat = (
        ndvi_tagged.dropna(subset=["phase_wheat"])
        .groupby(["field_id", "year", "phase_wheat"], as_index=False)["ndvi_mean"]
        .agg(ndvi_mean="mean", ndvi_max="max")
    )
    ndvi_wheat_wide = (
        ndvi_wheat.pivot_table(index=["field_id", "year"], columns="phase_wheat", values=["ndvi_mean", "ndvi_max"], aggfunc="first")
    )
    ndvi_wheat_wide.columns = [f"{a}_{b}" for a, b in ndvi_wheat_wide.columns.to_flat_index()]
    ndvi_wheat_wide = ndvi_wheat_wide.reset_index().rename(columns={
        "ndvi_mean_p1": "ndvi_wheat_p1_mean",
        "ndvi_mean_p2": "ndvi_wheat_p2_mean",
        "ndvi_mean_p3": "ndvi_wheat_p3_mean",
        "ndvi_max_p1": "ndvi_wheat_p1_max",
        "ndvi_max_p2": "ndvi_wheat_p2_max",
        "ndvi_max_p3": "ndvi_wheat_p3_max",
    })

    ndvi_sun = (
        ndvi_tagged.dropna(subset=["phase_sunflower"])
        .groupby(["field_id", "year", "phase_sunflower"], as_index=False)["ndvi_mean"]
        .agg(ndvi_mean="mean", ndvi_max="max")
    )
    ndvi_sun_wide = (
        ndvi_sun.pivot_table(index=["field_id", "year"], columns="phase_sunflower", values=["ndvi_mean", "ndvi_max"], aggfunc="first")
    )
    ndvi_sun_wide.columns = [f"{a}_{b}" for a, b in ndvi_sun_wide.columns.to_flat_index()]
    ndvi_sun_wide = ndvi_sun_wide.reset_index().rename(columns={
        "ndvi_mean_p1": "ndvi_sunflower_p1_mean",
        "ndvi_mean_p2": "ndvi_sunflower_p2_mean",
        "ndvi_mean_p3": "ndvi_sunflower_p3_mean",
        "ndvi_max_p1": "ndvi_sunflower_p1_max",
        "ndvi_max_p2": "ndvi_sunflower_p2_max",
        "ndvi_max_p3": "ndvi_sunflower_p3_max",
    })

    # -----------------------------
    # Merge to base
    # -----------------------------
    out = base2.merge(wx_phase_wide, on=["field_group_id", "year"], how="left")
    out = out.merge(ndvi_wheat_wide, on=["field_id", "year"], how="left")
    out = out.merge(ndvi_sun_wide, on=["field_id", "year"], how="left")

    wheat_cols = [c for c in out.columns if c.startswith("wx_wheat_") or c.startswith("ndvi_wheat_")]
    sun_cols = [c for c in out.columns if c.startswith("wx_sunflower_") or c.startswith("ndvi_sunflower_")]
    out.loc[~out["is_wheat"].fillna(False), wheat_cols] = np.nan
    out.loc[~out["is_sunflower"].fillna(False), sun_cols] = np.nan

    out = out.drop(columns=["standard_name", "is_wheat", "is_sunflower"], errors="ignore")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PHASES_DATASET_PATH, index=False)
    print(f"Saved phases dataset: {PHASES_DATASET_PATH}")
    print(f"   rows={len(out)} cols={len(out.columns)}")
    return PHASES_DATASET_PATH


def _train_one(
    *,
    dataset_path: Path,
    dataset_tag: str,
    subset_tag: str,
    crop_filter: Optional[str],
) -> dict:
    df = load_dataset(str(dataset_path), crop_filter=crop_filter)
    if df.empty:
        raise RuntimeError(f"Empty dataset after filtering: {dataset_tag}/{subset_tag} crop_filter={crop_filter}")

    X, y, groups, feature_cols = prepare_features(df, model_type="all", exclude_leaky=True)
    cv = GroupKFold(n_splits=5)
    pipe = build_pipeline_with_params(
        feature_cols,
        n_estimators=BEST_REG_GBDT_PARAMS["n_estimators"],
        max_depth=BEST_REG_GBDT_PARAMS["max_depth"],
        learning_rate=BEST_REG_GBDT_PARAMS["learning_rate"],
        min_samples_leaf=BEST_REG_GBDT_PARAMS["min_samples_leaf"],
        subsample=BEST_REG_GBDT_PARAMS["subsample"],
    )
    res = evaluate_with_oof(pipe, X, y, groups, cv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    oof_cols = [c for c in ["field_id", "year", "crop_id", "target_yield_t_ha"] if c in df.columns]
    oof_df = df[oof_cols].copy()
    oof_df["dataset_tag"] = dataset_tag
    oof_df["subset_tag"] = subset_tag
    oof_df["model_pred_oof_t_ha"] = res["oof_pred"]
    oof_df["model_pred_train_t_ha"] = res["train_pred"]
    oof_path = MODELS_DIR / f"oof_phases_{dataset_tag}_{subset_tag}.csv"
    oof_df.to_csv(oof_path, index=False)

    return {
        "dataset_tag": dataset_tag,
        "dataset_path": str(dataset_path),
        "subset_tag": subset_tag,
        "crop_filter": str(crop_filter) if crop_filter is not None else "",
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "cv_r2_mean": res["cv_r2_mean"],
        "cv_r2_std": res["cv_r2_std"],
        "cv_rmse_mean": res["cv_rmse_mean"],
        "cv_rmse_std": res["cv_rmse_std"],
        "cv_mae_mean": res["cv_mae_mean"],
        "cv_mae_std": res["cv_mae_std"],
        "cv_mape_mean": res["cv_mape_mean"],
        "cv_mape_std": res["cv_mape_std"],
        "train_r2": res["train_r2"],
        "train_rmse": res["train_rmse"],
        "train_mae": res["train_mae"],
        "train_mape": res["train_mape"],
        "oof_path": str(oof_path),
    }


def train_and_compare() -> Path:
    print("=" * 80)
    print("TRAIN BASELINE vs PHASES (GBDT, GroupKFold by year)")
    print("=" * 80)

    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Base dataset not found: {BASE_DATASET_PATH}")
    if not PHASES_DATASET_PATH.exists():
        raise FileNotFoundError(f"Phases dataset not found: {PHASES_DATASET_PATH}")

    runs = [
        ("allcrops", None),
        ("sunflower", STANDARD_NAME_SUNFLOWER),
        ("wheat", STANDARD_NAME_WHEAT_SPRING),
    ]

    rows: list[dict] = []
    for subset_tag, crop_filter in runs:
        rows.append(_train_one(
            dataset_path=BASE_DATASET_PATH,
            dataset_tag="baseline",
            subset_tag=subset_tag,
            crop_filter=crop_filter,
        ))
        rows.append(_train_one(
            dataset_path=PHASES_DATASET_PATH,
            dataset_tag="phases",
            subset_tag=subset_tag,
            crop_filter=crop_filter,
        ))

    comp = pd.DataFrame(rows)
    out_path = MODELS_DIR / "models_phases_comparison.csv"
    comp.to_csv(out_path, index=False)
    print(f"Saved comparison table: {out_path}")
    print()
    print(comp[[
        "subset_tag",
        "dataset_tag",
        "n_rows",
        "n_features",
        "cv_r2_mean",
        "cv_rmse_mean",
        "cv_mae_mean",
        "cv_mape_mean",
        "train_r2",
        "oof_path",
    ]].to_string(index=False))
    print()
    return out_path


def main():
    build_phase_dataset()
    train_and_compare()


if __name__ == "__main__":
    main()

