"""
Sprint 5.1 — Fetch Open-Meteo Historical Weather for all fields × all years.

API: https://archive-api.open-meteo.com/v1/archive
  - free, no API key, public
  - data source: ERA5 / ECMWF reanalysis (industry standard)
  - daily aggregates per (lat, lon, date)

We pull:
  - temperature_2m_max / min / mean
  - precipitation_sum, rain_sum, snowfall_sum
  - shortwave_radiation_sum (MJ/m²)         ← solar = photosynthesis driver
  - et0_fao_evapotranspiration (mm)         ← reference evapotranspiration
  - vapour_pressure_deficit_max (kPa)       ← drought/heat stress predictor
  - wind_speed_10m_max (km/h)

Output: data_raw/openmeteo_daily.csv  (long format: one row per field × date)
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
OUT_CSV = DATA_RAW / "openmeteo_daily.csv"

YEAR_START = 2010
YEAR_END = 2025

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit_max",
    "wind_speed_10m_max",
]

REQUEST_SLEEP_S = 6.0     # ERA5 archive endpoint is sensitive; be very polite
GRID_RESOLUTION = 0.1      # dedup fields to 0.1° (~10 km) — fields share an ERA5 cell anyway


def fetch_one_field(field_id: int, lat: float, lon: float, year_start: int, year_end: int) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": f"{year_start}-01-01",
        "end_date": f"{year_end}-12-31",
        "daily": ",".join(DAILY_PARAMS),
        "timezone": "auto",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "daily" not in data:
        raise RuntimeError(f"Unexpected response for field {field_id}: {data}")

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["field_id"] = int(field_id)
    df["om_lat_used"] = data.get("latitude")
    df["om_lon_used"] = data.get("longitude")
    df["om_elevation"] = data.get("elevation")
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    cols_order = (
        ["field_id", "date", "year", "doy"]
        + DAILY_PARAMS
        + ["om_lat_used", "om_lon_used", "om_elevation"]
    )
    return df[cols_order]


def main():
    print("=" * 80)
    print("FETCH OPEN-METEO HISTORICAL WEATHER")
    print("=" * 80)

    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "lat", "long"]].rename(columns={"id": "field_id"})
    fields = fields.dropna(subset=["lat", "long"]).copy()
    fields["field_id"] = fields["field_id"].astype(int)
    fields["lat"] = pd.to_numeric(fields["lat"], errors="coerce")
    fields["long"] = pd.to_numeric(fields["long"], errors="coerce")
    fields = fields.dropna(subset=["lat", "long"])

    # Dedup to grid cells (~10 km). Fields in the same cell share the same ERA5 weather.
    fields["lat_grid"] = (fields["lat"] / GRID_RESOLUTION).round() * GRID_RESOLUTION
    fields["lon_grid"] = (fields["long"] / GRID_RESOLUTION).round() * GRID_RESOLUTION
    unique_cells = fields.drop_duplicates(subset=["lat_grid", "lon_grid"])[["lat_grid", "lon_grid"]].reset_index(drop=True)
    print(f"Fetching for {len(fields)} fields → {len(unique_cells)} unique ERA5 cells (resolution {GRID_RESOLUTION}°)")
    print(f"  years {YEAR_START}..{YEAR_END}")
    print(f"  daily params: {DAILY_PARAMS}")
    print()

    cell_data: dict[tuple[float, float], pd.DataFrame] = {}
    failed: list[tuple[float, float]] = []
    for i, row in unique_cells.iterrows():
        lat_g = float(row["lat_grid"])
        lon_g = float(row["lon_grid"])
        attempts = 0
        df = None
        while attempts < 4:
            try:
                df = fetch_one_field(int(-1), lat_g, lon_g, YEAR_START, YEAR_END)
                break
            except Exception as e:
                attempts += 1
                wait = 15 * attempts
                print(f"  cell ({lat_g:.3f}, {lon_g:.3f}) attempt {attempts} failed: {e} — waiting {wait}s")
                time.sleep(wait)
        if df is None:
            failed.append((lat_g, lon_g))
            print(f"  [{i+1:2d}/{len(unique_cells)}] cell ({lat_g:.3f}, {lon_g:.3f})  GAVE UP")
            continue
        cell_data[(lat_g, lon_g)] = df
        print(f"  [{i+1:2d}/{len(unique_cells)}] cell ({lat_g:.3f}, {lon_g:.3f})  rows={len(df)}")
        time.sleep(REQUEST_SLEEP_S)

    if not cell_data:
        print("No data fetched.")
        return

    # Expand cell-level data to per-field rows
    parts: list[pd.DataFrame] = []
    for _, row in fields.iterrows():
        fid = int(row["field_id"])
        key = (float(row["lat_grid"]), float(row["lon_grid"]))
        if key not in cell_data:
            continue
        df_cell = cell_data[key].copy()
        df_cell["field_id"] = fid
        parts.append(df_cell)

    if not parts:
        print("No data fetched.")
        return

    out = pd.concat(parts, ignore_index=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print()
    print(f"Saved {OUT_CSV}")
    print(f"  total rows: {len(out)}")
    print(f"  fields with data: {out['field_id'].nunique()}")
    print(f"  date range: {out['date'].min()} .. {out['date'].max()}")
    if failed:
        print(f"  ⚠️ failed for fields: {failed}")

    print("\nQuick sanity check:")
    summary = (
        out.groupby("year")
        .agg(
            n=("date", "count"),
            t_min=("temperature_2m_min", "min"),
            t_max=("temperature_2m_max", "max"),
            precip_sum=("precipitation_sum", "sum"),
            srad_mean=("shortwave_radiation_sum", "mean"),
            vpd_max=("vapour_pressure_deficit_max", "max"),
        )
        .round(2)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
