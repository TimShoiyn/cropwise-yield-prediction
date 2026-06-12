"""
Sprint 5.2 — NDVI anomaly relative to per-field history.

For each (field_id, target_year, DOY):
    baseline(field_id, DOY) = mean NDVI for the same field_id, same DOY (±7 d window),
                              over years strictly < target_year (no leakage).
    anomaly(field_id, target_year, DOY) = NDVI(target_year, DOY) − baseline.

Then we aggregate anomaly up to each as-of date and add columns:
    ndvi_anom_mean_asof, ndvi_anom_max_asof, ndvi_anom_min_asof,
    ndvi_anom_last_value_asof, ndvi_anom_last30d_mean_asof,
    ndvi_baseline_n_years    (how many years of history were available)

Output: data_processed/ml_dataset_clean_v4_asof_{tag}.csv  (= v3 + anomaly cols)
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

NDVI_PATH = DATA_RAW / "ndvi_timeseries.csv"
ASOF_TAGS = ["07_01", "08_01", "09_01"]
DOY_WINDOW = 7   # ±7 days when computing baseline (smooths noise)


def load_ndvi() -> pd.DataFrame:
    df = pd.read_csv(NDVI_PATH, usecols=["field_id", "year", "date", "ndvi_mean", "cloud_coverage", "data_coverage"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ndvi_mean"]).copy()
    if "cloud_coverage" in df.columns:
        df = df[df["cloud_coverage"].fillna(0) <= 30]   # filter cloudy
    df["year"] = df["date"].dt.year.astype(int)
    df["doy"] = df["date"].dt.dayofyear.astype(int)
    df["field_id"] = df["field_id"].astype(int)
    df = df.sort_values(["field_id", "year", "date"]).reset_index(drop=True)
    return df


def build_baseline_for_year(ndvi: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Returns DataFrame [field_id, doy, ndvi_baseline, n_years]."""
    hist = ndvi[ndvi["year"] < target_year]
    if hist.empty:
        return pd.DataFrame(columns=["field_id", "doy", "ndvi_baseline", "n_years"])
    # Pre-aggregate to (field_id, year, doy) mean (multiple obs per day across sources)
    daily = hist.groupby(["field_id", "year", "doy"])["ndvi_mean"].mean().reset_index()
    # Rolling DOY window ±7
    rows = []
    for fid, g in daily.groupby("field_id"):
        # Build a year×doy matrix
        for doy_center in range(1, 367):
            mask = (g["doy"] >= doy_center - DOY_WINDOW) & (g["doy"] <= doy_center + DOY_WINDOW)
            sel = g[mask]
            if sel.empty:
                continue
            mean_by_year = sel.groupby("year")["ndvi_mean"].mean()
            if len(mean_by_year) == 0:
                continue
            rows.append({
                "field_id": int(fid),
                "doy": int(doy_center),
                "ndvi_baseline": float(mean_by_year.mean()),
                "n_years": int(len(mean_by_year)),
            })
    return pd.DataFrame(rows)


def aggregate_anomaly(ndvi_year: pd.DataFrame, baseline: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if ndvi_year.empty or baseline.empty:
        return pd.DataFrame()
    df = ndvi_year[ndvi_year["date"] < as_of].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.merge(baseline, on=["field_id", "doy"], how="left")
    df["anomaly"] = df["ndvi_mean"] - df["ndvi_baseline"]

    last30_threshold = as_of - pd.Timedelta(days=30)
    out_rows = []
    for fid, g in df.groupby("field_id"):
        g = g.sort_values("date")
        anom = g["anomaly"].dropna().values
        if anom.size == 0:
            continue
        last30 = g[g["date"] >= last30_threshold]
        last30_mean = float(np.nanmean(last30["anomaly"])) if not last30.empty else np.nan
        out_rows.append({
            "field_id": int(fid),
            "year": int(g["year"].iloc[0]),
            "ndvi_anom_mean_asof": float(np.nanmean(anom)),
            "ndvi_anom_max_asof": float(np.nanmax(anom)),
            "ndvi_anom_min_asof": float(np.nanmin(anom)),
            "ndvi_anom_last_value_asof": float(g.dropna(subset=["anomaly"])["anomaly"].iloc[-1]) if g["anomaly"].notna().any() else np.nan,
            "ndvi_anom_last30d_mean_asof": last30_mean,
            "ndvi_baseline_n_years": int(g["n_years"].max()) if "n_years" in g.columns else 0,
        })
    return pd.DataFrame(out_rows)


def build_for_asof(asof_tag: str, ndvi: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v3_asof_{asof_tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{asof_tag}.csv"
    base = pd.read_csv(in_path)
    print(f"\nas-of {asof_tag} — base: {len(base)} rows, {base.shape[1]} cols")

    parts = []
    target_years = sorted(base["year"].dropna().astype(int).unique())
    for ty in target_years:
        baseline = build_baseline_for_year(ndvi, ty)
        ndvi_year = ndvi[ndvi["year"] == ty]
        mm, dd = int(asof_tag.split("_")[0]), int(asof_tag.split("_")[1])
        as_of = pd.Timestamp(year=ty, month=mm, day=dd)
        anom = aggregate_anomaly(ndvi_year, baseline, as_of)
        if not anom.empty:
            parts.append(anom)

    if not parts:
        print("  no anomaly data computed"); return

    anomalies = pd.concat(parts, ignore_index=True)
    out = base.merge(anomalies, on=["field_id", "year"], how="left")
    out.to_csv(out_path, index=False)
    print(f"  output: {out_path.name} → {len(out)} rows × {out.shape[1]} cols")
    new_cols = [c for c in out.columns if c.startswith("ndvi_anom_") or c == "ndvi_baseline_n_years"]
    coverage = out[new_cols].notna().mean()
    print(f"  new feature coverage:")
    for c, v in coverage.items():
        print(f"    {c}: {v*100:.1f}%")


def main():
    print("=" * 80)
    print("BUILD NDVI ANOMALY (Sprint 5.2)")
    print("=" * 80)
    print(f"Loading {NDVI_PATH}...")
    ndvi = load_ndvi()
    print(f"  {len(ndvi)} valid NDVI obs, {ndvi['field_id'].nunique()} fields, years {ndvi['year'].min()}..{ndvi['year'].max()}")

    for tag in ASOF_TAGS:
        build_for_asof(tag, ndvi)

    print("\nDone.")


if __name__ == "__main__":
    main()
