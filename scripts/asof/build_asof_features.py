"""
Sprint 3.1 — As-of feature dataset builder.

Goal: simulate "what does the model know at date D?" for in-season forecasting.

Input:
  - data_processed/ml_dataset_clean_v2.csv  (target + soil/static features)
  - data_raw/ndvi_timeseries.csv            (per-day NDVI per field)
  - data_raw/weather_history_items.csv      (per-day weather per field_group)
  - data_raw/fields.csv                     (field_id -> field_group_id)

For each `as_of` date in {07-01, 08-01, 09-01}, for each (field_id, year) row:
  - Recompute NDVI aggregates using only NDVI observations with date <= year-as_of
  - Recompute weather aggregates using only weather rows with date <= year-as_of
  - Keep static features (soil, crop_id, prev_crop_id, field_id, area)
  - Drop the original full-season NDVI/weather and phase features (they leak post-as_of info)

Output:
  - data_processed/ml_dataset_clean_v2_asof_{MM_DD}.csv
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")

CLEAN_DATASET = DATA_PROCESSED / "ml_dataset_clean_v2.csv"
NDVI_CSV = DATA_RAW / "ndvi_timeseries.csv"
WX_CSV = DATA_RAW / "weather_history_items.csv"
FIELDS_CSV = DATA_RAW / "fields.csv"

# Season window for aggregation. Even though we cut at as_of, we still need a
# start date so we don't pull NDVI from previous season's residual.
SEASON_START_MONTH = 4   # April 1
SEASON_START_DAY = 1

AS_OF_DATES = [
    ("07_01", 7, 1),
    ("08_01", 8, 1),
    ("09_01", 9, 1),
]

# Columns to drop from the v2 dataset because they encode full-season info (post-as_of leakage)
LEAKY_FULL_SEASON_COLS = [
    "ndvi_observations",
    "ndvi_mean_season",
    "ndvi_max_season",
    "ndvi_early",
    "ndvi_mid",
    "ndvi_late",
    "weather_temp_avg_season",
    "weather_gdd_season",
    "weather_precip_sum_season",
    "weather_precip_sum_early",
    "weather_hot_days",
    # Phase features built from the full-season weather/NDVI:
    "wx_wheat_p1_precip_sum",
    "wx_wheat_p2_precip_sum",
    "wx_wheat_p3_precip_sum",
    "wx_sunflower_p1_hot_days",
    "wx_sunflower_p2_hot_days",
    "wx_sunflower_p3_hot_days",
    "ndvi_wheat_p1_max",
    "ndvi_wheat_p2_max",
    "ndvi_wheat_p3_max",
    "ndvi_wheat_p1_mean",
    "ndvi_wheat_p2_mean",
    "ndvi_wheat_p3_mean",
    "ndvi_sunflower_p1_max",
    "ndvi_sunflower_p2_max",
    "ndvi_sunflower_p3_max",
    "ndvi_sunflower_p1_mean",
    "ndvi_sunflower_p2_mean",
    "ndvi_sunflower_p3_mean",
]


def _ensure_datetime(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def compute_ndvi_features_until(ndvi_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate NDVI per (field_id, year) for dates in [season_start, as_of_date].

    Returns columns:
      ndvi_n_obs_asof, ndvi_mean_asof, ndvi_max_asof, ndvi_min_asof,
      ndvi_std_asof, ndvi_p25_asof, ndvi_p75_asof,
      ndvi_p10_max_asof  (mean of top-10% NDVI = robust max),
      ndvi_slope_asof    (linear slope vs day-of-year),
      ndvi_last_value_asof
    """
    season_start = pd.Timestamp(year=as_of_date.year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    sub = ndvi_df[(ndvi_df["date"] >= season_start) & (ndvi_df["date"] <= as_of_date)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["doy"] = sub["date"].dt.dayofyear
    grouped = sub.groupby(["field_id", "year"], as_index=False)

    out_rows = []
    for (fid, yr), g in grouped:
        n = len(g)
        if n == 0:
            continue
        v = g["ndvi_mean"].astype(float).values
        d = g["doy"].astype(float).values
        # Slope: NDVI vs DOY
        if n >= 2 and np.std(d) > 0:
            slope = np.polyfit(d, v, 1)[0]
        else:
            slope = np.nan
        # Top-10% mean
        k = max(1, n // 10)
        top_k_mean = np.mean(np.sort(v)[-k:])
        out_rows.append(
            {
                "field_id": int(fid),
                "year": int(yr),
                "ndvi_n_obs_asof": int(n),
                "ndvi_mean_asof": float(np.mean(v)),
                "ndvi_max_asof": float(np.max(v)),
                "ndvi_min_asof": float(np.min(v)),
                "ndvi_std_asof": float(np.std(v)),
                "ndvi_p25_asof": float(np.quantile(v, 0.25)),
                "ndvi_p75_asof": float(np.quantile(v, 0.75)),
                "ndvi_p10_max_asof": float(top_k_mean),
                "ndvi_slope_asof": float(slope) if pd.notna(slope) else np.nan,
                "ndvi_last_value_asof": float(g.sort_values("date")["ndvi_mean"].iloc[-1]),
            }
        )
    return pd.DataFrame(out_rows)


def compute_weather_features_until(
    wx_df: pd.DataFrame, fields_groups: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.DataFrame:
    """Aggregate weather per (field_group_id, year). Then we'll join via fields_groups."""
    season_start = pd.Timestamp(year=as_of_date.year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    sub = wx_df[(wx_df["date"] >= season_start) & (wx_df["date"] <= as_of_date)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["gdd_t5"] = np.maximum(sub["temperature_avg"] - 5.0, 0.0)
    sub["gdd_t6"] = np.maximum(sub["temperature_avg"] - 6.0, 0.0)
    sub["hot_d30"] = (sub["temperature_max"] > 30.0).astype(float)
    sub["hot_d35"] = (sub["temperature_max"] > 35.0).astype(float)
    sub["dry_day"] = (sub["precipitation"] < 1.0).astype(float)

    # Early-season window: April 1 — June 1 (already ≤ as_of_date in summer asofs)
    early_end = pd.Timestamp(year=as_of_date.year, month=6, day=1)
    sub_early = sub[sub["date"] <= early_end]

    agg = (
        sub.groupby(["field_group_id", "year"])
        .agg(
            wx_temp_avg_asof=("temperature_avg", "mean"),
            wx_temp_max_asof=("temperature_max", "max"),
            wx_precip_sum_asof=("precipitation", "sum"),
            wx_gdd_t5_asof=("gdd_t5", "sum"),
            wx_gdd_t6_asof=("gdd_t6", "sum"),
            wx_hot_d30_asof=("hot_d30", "sum"),
            wx_hot_d35_asof=("hot_d35", "sum"),
            wx_dry_days_asof=("dry_day", "sum"),
        )
        .reset_index()
    )

    early_agg = (
        sub_early.groupby(["field_group_id", "year"])
        .agg(
            wx_precip_sum_early_asof=("precipitation", "sum"),
            wx_hot_d30_early_asof=("hot_d30", "sum"),
            wx_dry_days_early_asof=("dry_day", "sum"),
        )
        .reset_index()
    )

    out = agg.merge(early_agg, on=["field_group_id", "year"], how="left")
    # Now expand to (field_id, year) using fields_groups
    out = out.merge(fields_groups, on="field_group_id", how="left")
    out = out.dropna(subset=["field_id"])
    out["field_id"] = out["field_id"].astype(int)
    out["year"] = out["year"].astype(int)
    out = out.drop(columns=["field_group_id"])
    return out


def build_for_asof(base: pd.DataFrame, ndvi: pd.DataFrame, wx: pd.DataFrame, fields_g: pd.DataFrame, mm: int, dd: int, tag: str) -> Path:
    print(f"\n--- as_of {mm:02d}-{dd:02d} ---")

    # We need to compute features per year (each row's year is a different actual as_of date)
    rows_out = []
    for yr, year_rows in base.groupby("year"):
        as_of_ts = pd.Timestamp(year=int(yr), month=mm, day=dd)
        ndvi_year = compute_ndvi_features_until(ndvi[ndvi["year"] == int(yr)], as_of_ts)
        wx_year = compute_weather_features_until(wx[wx["year"] == int(yr)], fields_g, as_of_ts)

        merged = year_rows
        if not ndvi_year.empty:
            merged = merged.merge(ndvi_year, on=["field_id", "year"], how="left")
        if not wx_year.empty:
            merged = merged.merge(wx_year, on=["field_id", "year"], how="left")
        rows_out.append(merged)

    out = pd.concat(rows_out, ignore_index=True)

    # Drop full-season leaky features (we rebuilt as-of versions)
    drop_cols = [c for c in LEAKY_FULL_SEASON_COLS if c in out.columns]
    out = out.drop(columns=drop_cols)

    out_path = DATA_PROCESSED / f"ml_dataset_clean_v2_asof_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  rows={len(out)} cols={len(out.columns)}")

    # NaN summary for the new asof features
    new_cols = [c for c in out.columns if c.endswith("_asof")]
    nan_pct = (out[new_cols].isna().sum() / len(out) * 100).round(1)
    print("  NaN % for asof features:")
    for c, v in nan_pct.items():
        print(f"    {c:30s} {v:5.1f}%")
    return out_path


def main():
    print("=" * 80)
    print("BUILD AS-OF DATASETS — Sprint 3.1")
    print("=" * 80)

    base = pd.read_csv(CLEAN_DATASET)
    print(f"Base v2: {len(base)} rows × {len(base.columns)} cols")

    ndvi = pd.read_csv(NDVI_CSV)
    ndvi["date"] = _ensure_datetime(ndvi["date"])
    ndvi["field_id"] = pd.to_numeric(ndvi["field_id"], errors="coerce")
    ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce")
    ndvi["ndvi_mean"] = pd.to_numeric(ndvi["ndvi_mean"], errors="coerce")
    ndvi = ndvi.dropna(subset=["field_id", "year", "date", "ndvi_mean"]).copy()
    ndvi["field_id"] = ndvi["field_id"].astype(int)
    ndvi["year"] = ndvi["year"].astype(int)
    print(f"NDVI: {len(ndvi)} obs over {ndvi['field_id'].nunique()} fields")

    wx = pd.read_csv(WX_CSV)
    wx["date"] = _ensure_datetime(wx["date"])
    wx["year"] = pd.to_numeric(wx["year"], errors="coerce").astype("Int64")
    for c in ["temperature_avg", "temperature_max", "precipitation"]:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx["field_group_id"] = pd.to_numeric(wx["field_group_id"], errors="coerce")
    wx = wx.dropna(subset=["date", "field_group_id", "year"]).copy()
    print(f"Weather: {len(wx)} day-rows for {wx['field_group_id'].nunique()} groups")

    fields_g = (
        pd.read_csv(FIELDS_CSV)[["id", "field_group_id"]]
        .rename(columns={"id": "field_id"})
        .dropna()
    )
    fields_g["field_id"] = fields_g["field_id"].astype(int)
    fields_g["field_group_id"] = pd.to_numeric(fields_g["field_group_id"], errors="coerce").astype(int)

    for tag, mm, dd in AS_OF_DATES:
        build_for_asof(base, ndvi, wx, fields_g, mm, dd, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
