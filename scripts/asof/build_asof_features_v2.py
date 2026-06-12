"""
Sprint 3.2 — Add agronomic features to as-of datasets.

New features added on top of `ml_dataset_clean_v2_asof_{tag}.csv`:

NDVI (per (field_id, year) up to as_of):
  - ndvi_integral_asof     : sum of NDVI values
  - ndvi_peak_doy_asof     : day-of-year of max NDVI
  - ndvi_amplitude_asof    : max - min
  - ndvi_above_05_days_asof: count of days with NDVI > 0.5 (proxy for active vegetation length)

Weather (per (field_group_id, year) up to as_of):
  - wx_drought_run_asof    : longest consecutive run of dry days (precip<1mm)
  - wx_heat_run_asof       : longest run of hot days (Tmax>30C)
  - wx_jul_precip_asof     : precip sum in July up to as_of (NaN if as_of < Jul)
  - wx_jul_hot_days_asof   : hot days in July up to as_of
  - wx_aug_precip_asof     : precip sum in August up to as_of
  - wx_aug_hot_days_asof   : hot days in August up to as_of

Field history (independent of as_of):
  - years_since_sunflower  : seasons since this field was sunflower (>=1; 99 if never seen)
  - years_since_wheat      : same for wheat (any kind)
  - rotation_pair          : "{prev_crop_id}__{crop_id}" (categorical)

Usage:
  python scripts/asof/build_asof_features_v2.py
Outputs:
  - data_processed/ml_dataset_clean_v2b_asof_{tag}.csv  (07_01, 08_01, 09_01)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")

NDVI_CSV = DATA_RAW / "ndvi_timeseries.csv"
WX_CSV = DATA_RAW / "weather_history_items.csv"
FIELDS_CSV = DATA_RAW / "fields.csv"
HIST_CSV = DATA_RAW / "history_items_full.csv"

AS_OF_TAGS = [("07_01", 7, 1), ("08_01", 8, 1), ("09_01", 9, 1)]

SEASON_START_MONTH = 4
SEASON_START_DAY = 1


def _ensure_dt(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def ndvi_v2_features(ndvi: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    season_start = pd.Timestamp(year=as_of.year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    sub = ndvi[(ndvi["date"] >= season_start) & (ndvi["date"] <= as_of)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["doy"] = sub["date"].dt.dayofyear

    rows = []
    for (fid, yr), g in sub.groupby(["field_id", "year"]):
        v = g["ndvi_mean"].astype(float).values
        d = g["doy"].astype(int).values
        if len(v) == 0:
            continue
        peak_idx = int(np.argmax(v))
        rows.append(
            {
                "field_id": int(fid),
                "year": int(yr),
                "ndvi_integral_asof": float(np.sum(v)),
                "ndvi_peak_doy_asof": int(d[peak_idx]),
                "ndvi_amplitude_asof": float(np.max(v) - np.min(v)),
                "ndvi_above_05_days_asof": int((v > 0.5).sum()),
            }
        )
    return pd.DataFrame(rows)


def _longest_run_of_ones(arr: np.ndarray) -> int:
    if len(arr) == 0:
        return 0
    best = cur = 0
    for v in arr:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def wx_v2_features(wx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    season_start = pd.Timestamp(year=as_of.year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    sub = wx[(wx["date"] >= season_start) & (wx["date"] <= as_of)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(["field_group_id", "date"])
    sub["dry"] = (sub["precipitation"] < 1.0).astype(int)
    sub["hot"] = (sub["temperature_max"] > 30.0).astype(int)

    sub["month"] = sub["date"].dt.month

    rows = []
    for (fg, yr), g in sub.groupby(["field_group_id", "year"]):
        rows.append(
            {
                "field_group_id": int(fg),
                "year": int(yr),
                "wx_drought_run_asof": _longest_run_of_ones(g["dry"].values),
                "wx_heat_run_asof": _longest_run_of_ones(g["hot"].values),
                "wx_jul_precip_asof": float(g[g["month"] == 7]["precipitation"].sum()),
                "wx_jul_hot_days_asof": float(g[g["month"] == 7]["hot"].sum()),
                "wx_aug_precip_asof": float(g[g["month"] == 8]["precipitation"].sum()),
                "wx_aug_hot_days_asof": float(g[g["month"] == 8]["hot"].sum()),
            }
        )
    return pd.DataFrame(rows)


def field_history_features(crops_df: pd.DataFrame) -> pd.DataFrame:
    """
    From history_items_full, for each (field_id, year):
      - years_since_sunflower
      - years_since_wheat (any wheat)
      - rotation_pair  (constructed externally; we just compute the year-since metrics)
    """
    hi = pd.read_csv(HIST_CSV, low_memory=False)[["field_id", "year", "crop_id"]].dropna()
    hi["field_id"] = hi["field_id"].astype(int)
    hi["year"] = hi["year"].astype(int)
    hi["crop_id"] = hi["crop_id"].astype(int)
    hi = hi.merge(crops_df, on="crop_id", how="left")
    hi = hi.sort_values(["field_id", "year"])

    SUNFLOWER = "sunflower"
    WHEAT_SET = {"wheat_spring", "wheat_winter", "rye_winter"}

    out_rows = []
    for fid, g in hi.groupby("field_id"):
        seen_sunf = -1
        seen_wheat = -1
        for _, row in g.iterrows():
            yr = int(row["year"])
            ys_sun = (yr - seen_sunf) if seen_sunf > 0 else 99
            ys_wh = (yr - seen_wheat) if seen_wheat > 0 else 99
            out_rows.append(
                {
                    "field_id": int(fid),
                    "year": yr,
                    "years_since_sunflower": int(ys_sun),
                    "years_since_wheat": int(ys_wh),
                }
            )
            std = row.get("standard_name")
            if std == SUNFLOWER:
                seen_sunf = yr
            if std in WHEAT_SET:
                seen_wheat = yr
    return pd.DataFrame(out_rows).drop_duplicates(subset=["field_id", "year"])


def main():
    print("=" * 80)
    print("BUILD AS-OF v2 (agronomic features) — Sprint 3.2")
    print("=" * 80)

    crops_df = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})

    print("\nLoading NDVI...")
    ndvi = pd.read_csv(NDVI_CSV)
    ndvi["date"] = _ensure_dt(ndvi["date"])
    ndvi["field_id"] = pd.to_numeric(ndvi["field_id"], errors="coerce")
    ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce")
    ndvi["ndvi_mean"] = pd.to_numeric(ndvi["ndvi_mean"], errors="coerce")
    ndvi = ndvi.dropna(subset=["field_id", "year", "date", "ndvi_mean"])
    ndvi["field_id"] = ndvi["field_id"].astype(int)
    ndvi["year"] = ndvi["year"].astype(int)

    print("Loading weather...")
    wx = pd.read_csv(WX_CSV)
    wx["date"] = _ensure_dt(wx["date"])
    wx["year"] = pd.to_numeric(wx["year"], errors="coerce")
    wx["field_group_id"] = pd.to_numeric(wx["field_group_id"], errors="coerce")
    for c in ["temperature_avg", "temperature_max", "precipitation"]:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx = wx.dropna(subset=["date", "year", "field_group_id"])
    wx["year"] = wx["year"].astype(int)
    wx["field_group_id"] = wx["field_group_id"].astype(int)

    print("Loading fields...")
    fields_g = pd.read_csv(FIELDS_CSV)[["id", "field_group_id"]].rename(columns={"id": "field_id"}).dropna()
    fields_g["field_id"] = fields_g["field_id"].astype(int)
    fields_g["field_group_id"] = pd.to_numeric(fields_g["field_group_id"], errors="coerce").astype(int)

    print("Building field history features...")
    hist_feats = field_history_features(crops_df)
    print(f"  history rows: {len(hist_feats)}")

    for tag, mm, dd in AS_OF_TAGS:
        print(f"\n--- as_of {tag} ---")
        in_path = DATA_PROCESSED / f"ml_dataset_clean_v2_asof_{tag}.csv"
        if not in_path.exists():
            print(f"  Missing: {in_path}, skipping")
            continue
        base = pd.read_csv(in_path)
        base = base.merge(crops_df, on="crop_id", how="left")
        # Attach field_group_id to base (needed for wx join). Will be dropped at end.
        if "field_group_id" not in base.columns:
            base = base.merge(fields_g, on="field_id", how="left")

        # Per-year compute of NDVI v2 + WX v2 features
        out_parts = []
        for yr, year_rows in base.groupby("year"):
            as_of_ts = pd.Timestamp(year=int(yr), month=mm, day=dd)
            n_feat = ndvi_v2_features(ndvi[ndvi["year"] == int(yr)], as_of_ts)
            w_feat = wx_v2_features(wx[wx["year"] == int(yr)], as_of_ts)
            merged = year_rows.copy()
            if not n_feat.empty:
                merged = merged.merge(n_feat, on=["field_id", "year"], how="left")
            if not w_feat.empty and "field_group_id" in merged.columns:
                merged = merged.merge(w_feat, on=["field_group_id", "year"], how="left")
            out_parts.append(merged)

        out = pd.concat(out_parts, ignore_index=True)

        # Add field history features
        out = out.merge(hist_feats, on=["field_id", "year"], how="left")
        out["years_since_sunflower"] = out["years_since_sunflower"].fillna(99).astype(int)
        out["years_since_wheat"] = out["years_since_wheat"].fillna(99).astype(int)

        # Build rotation_pair as categorical string
        out["rotation_pair"] = (
            out["prev_crop_id"].fillna(-1).astype(int).astype(str)
            + "__"
            + out["crop_id"].fillna(-1).astype(int).astype(str)
        )

        # Drop helper join cols if any
        out = out.drop(columns=["standard_name"], errors="ignore")

        # Drop field_group_id (we used it only for joining wx)
        if "field_group_id" in out.columns:
            out = out.drop(columns=["field_group_id"])

        out_path = DATA_PROCESSED / f"ml_dataset_clean_v2b_asof_{tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  rows={len(out)} cols={len(out.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
