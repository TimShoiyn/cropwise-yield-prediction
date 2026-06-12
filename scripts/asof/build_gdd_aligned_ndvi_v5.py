"""
Sprint 5.3 — GDD-aligned NDVI features.

Idea: re-sample each (field × year) NDVI series along an accumulated-GDD axis
(base 5°C, accumulated from April 1) instead of the calendar axis. This makes
years biophysically comparable: NDVI at 600 GDD is the same physiological
stage in 2018 (cool) and 2022 (hot), even if the calendar dates differ.

Features (one per stage; NaN if as-of GDD has not reached the stage yet):
    ndvi_at_gdd_200, ndvi_at_gdd_400, ndvi_at_gdd_600,
    ndvi_at_gdd_800, ndvi_at_gdd_1000, ndvi_at_gdd_1200, ndvi_at_gdd_1400
    gdd_at_asof              (current accumulated GDD)
    ndvi_peak_gdd_at_asof    (GDD at the date of season-to-date peak NDVI)
    ndvi_growth_per_100gdd   (slope from emergence stage to current peak)

Output: data_processed/ml_dataset_clean_v5_asof_{tag}.csv
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

OM_PATH = DATA_RAW / "openmeteo_daily.csv"
NDVI_PATH = DATA_RAW / "ndvi_timeseries.csv"
ASOF_TAGS = ["07_01", "08_01", "09_01"]

GDD_BASE = 5.0
GDD_STAGES = [200, 400, 600, 800, 1000, 1200, 1400]
NDVI_PEAK_MIN = 0.30   # below this we don't trust "growth" slope


def build_gdd_track(om: pd.DataFrame) -> pd.DataFrame:
    """Cumulative GDD from April 1 per (field_id, year, date)."""
    out = []
    for (fid, yr), g in om.groupby(["field_id", "year"]):
        g = g.sort_values("date").copy()
        apr1 = pd.Timestamp(year=int(yr), month=4, day=1)
        g = g[g["date"] >= apr1]
        if g.empty:
            continue
        gdd_daily = np.clip(g["temperature_2m_mean"].fillna(0) - GDD_BASE, 0, None)
        g["gdd"] = gdd_daily.cumsum().values
        out.append(g[["field_id", "year", "date", "gdd"]])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def smooth_ndvi(ndvi: pd.DataFrame) -> pd.DataFrame:
    """Per (field_id, year), 7-day rolling mean on daily-aggregated NDVI."""
    daily = (
        ndvi.groupby(["field_id", "year", "date"])["ndvi_mean"]
        .mean().reset_index()
        .sort_values(["field_id", "year", "date"])
    )
    smoothed = []
    for (fid, yr), g in daily.groupby(["field_id", "year"]):
        g = g.sort_values("date").copy()
        g["ndvi_smooth"] = g["ndvi_mean"].rolling(7, center=True, min_periods=2).mean()
        smoothed.append(g)
    return pd.concat(smoothed, ignore_index=True)


def interp_ndvi_at_gdd(g: pd.DataFrame, target_gdd: float) -> float:
    """g must have columns: gdd (sorted asc), ndvi_smooth. NaN if target > max GDD."""
    if g.empty or target_gdd > g["gdd"].max() or target_gdd < g["gdd"].min():
        return float("nan")
    # numpy interp requires non-decreasing xp
    xp = g["gdd"].values
    fp = g["ndvi_smooth"].values
    mask = ~np.isnan(fp)
    if mask.sum() < 2:
        return float("nan")
    return float(np.interp(target_gdd, xp[mask], fp[mask]))


def features_for_asof(ndvi_smooth: pd.DataFrame, gdd_track: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    year = as_of.year
    ndvi_y = ndvi_smooth[(ndvi_smooth["year"] == year) & (ndvi_smooth["date"] < as_of)].copy()
    gdd_y = gdd_track[(gdd_track["year"] == year) & (gdd_track["date"] < as_of)].copy()
    if ndvi_y.empty or gdd_y.empty:
        return pd.DataFrame()
    merged = ndvi_y.merge(gdd_y[["field_id", "date", "gdd"]], on=["field_id", "date"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged = merged.sort_values(["field_id", "gdd"])

    out_rows = []
    for fid, g in merged.groupby("field_id"):
        g = g.sort_values("gdd").reset_index(drop=True)
        gdd_at_asof = float(g["gdd"].max())
        row: dict = {"field_id": int(fid), "year": int(year), "gdd_at_asof": gdd_at_asof}
        for stage in GDD_STAGES:
            row[f"ndvi_at_gdd_{stage}"] = interp_ndvi_at_gdd(g, stage) if stage <= gdd_at_asof else float("nan")
        # Peak NDVI GDD position
        ns = g["ndvi_smooth"]
        if ns.notna().any():
            peak_idx = ns.idxmax()
            row["ndvi_peak_gdd_at_asof"] = float(g.loc[peak_idx, "gdd"])
            row["ndvi_peak_value_at_asof"] = float(ns.max())
        else:
            row["ndvi_peak_gdd_at_asof"] = float("nan")
            row["ndvi_peak_value_at_asof"] = float("nan")
        # Growth rate per 100 GDD: from first reliable point to peak
        emer = g[(g["ndvi_smooth"] >= 0.20) & g["ndvi_smooth"].notna()]
        if not emer.empty and ns.notna().any():
            g0 = float(emer.iloc[0]["gdd"]); n0 = float(emer.iloc[0]["ndvi_smooth"])
            g_pk = float(g.loc[peak_idx, "gdd"]); n_pk = float(ns.max())
            if g_pk - g0 > 50:
                row["ndvi_growth_per_100gdd"] = (n_pk - n0) / (g_pk - g0) * 100.0
            else:
                row["ndvi_growth_per_100gdd"] = float("nan")
        else:
            row["ndvi_growth_per_100gdd"] = float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def build_for_asof(asof_tag: str, ndvi_smooth: pd.DataFrame, gdd_track: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{asof_tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v5_asof_{asof_tag}.csv"
    base = pd.read_csv(in_path)
    print(f"\nas-of {asof_tag} — base v4: {len(base)} rows × {base.shape[1]} cols")

    parts = []
    for ty in sorted(base["year"].dropna().astype(int).unique()):
        mm, dd = int(asof_tag.split("_")[0]), int(asof_tag.split("_")[1])
        as_of = pd.Timestamp(year=ty, month=mm, day=dd)
        feat = features_for_asof(ndvi_smooth, gdd_track, as_of)
        if not feat.empty:
            parts.append(feat)

    if not parts:
        print("  no features computed"); return

    feats = pd.concat(parts, ignore_index=True)
    out = base.merge(feats, on=["field_id", "year"], how="left")
    out.to_csv(out_path, index=False)
    print(f"  output: {out_path.name} → {len(out)} rows × {out.shape[1]} cols")
    new_cols = [c for c in feats.columns if c not in ("field_id", "year")]
    cov = out[new_cols].notna().mean()
    print(f"  new feature coverage:")
    for c, v in cov.items():
        print(f"    {c}: {v*100:.1f}%")


def main():
    print("=" * 80)
    print("BUILD GDD-ALIGNED NDVI (Sprint 5.3)")
    print("=" * 80)
    print(f"Loading {OM_PATH}...")
    om = pd.read_csv(OM_PATH, usecols=["field_id", "date", "year", "temperature_2m_mean"])
    om["date"] = pd.to_datetime(om["date"], errors="coerce")
    om["field_id"] = om["field_id"].astype(int)
    print(f"  {len(om)} weather days, {om['field_id'].nunique()} fields")

    print(f"Loading {NDVI_PATH}...")
    ndvi = pd.read_csv(NDVI_PATH, usecols=["field_id", "year", "date", "ndvi_mean", "cloud_coverage"])
    ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
    ndvi = ndvi.dropna(subset=["date", "ndvi_mean"]).copy()
    if "cloud_coverage" in ndvi.columns:
        ndvi = ndvi[ndvi["cloud_coverage"].fillna(0) <= 30]
    ndvi["year"] = ndvi["date"].dt.year.astype(int)
    ndvi["field_id"] = ndvi["field_id"].astype(int)
    print(f"  {len(ndvi)} cloud-clean NDVI obs")

    print("Building cumulative GDD tracks...")
    gdd_track = build_gdd_track(om)
    print(f"  {len(gdd_track)} (field × date) GDD points")

    print("Smoothing NDVI...")
    ndvi_smooth = smooth_ndvi(ndvi)
    print(f"  {len(ndvi_smooth)} smoothed daily NDVI rows")

    for tag in ASOF_TAGS:
        build_for_asof(tag, ndvi_smooth, gdd_track)

    print("\nDone.")


if __name__ == "__main__":
    main()
