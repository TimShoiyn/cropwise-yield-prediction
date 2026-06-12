"""
B1: resumable Sentinel-2 red-edge/EVI/GCVI extractor.

Why:
  Cropwise historical_values has NDVI/temperature/soil_moisture, but not NDRE,
  EVI or GCVI. Literature suggests these indices can improve wheat late-season
  and dense-canopy performance where NDVI saturates.

Data source:
  Microsoft Planetary Computer STAC, Sentinel-2 L2A COGs.

Feature policy:
  For each (field_id, year, asof_tag), pick the least-cloudy Sentinel-2 scene
  in the 30 days before the as-of date and compute field-level mean:
    - NDVI  (B08, B04) at 10m, for sanity alignment;
    - EVI   (B08, B04, B02) at 10m;
    - GCVI  (B08, B03) at 10m;
    - NDRE  (B8A, B05) at 20m.

Output:
  data_raw/sentinel2_indices_b1.csv

Usage:
  # smoke/sample
  python3 scripts/external/fetch_sentinel2_indices_b1.py --limit 20

  # continue/resume full extraction
  python3 scripts/external/fetch_sentinel2_indices_b1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.mask import mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
OUT_CSV = DATA_RAW / "sentinel2_indices_b1.csv"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
ASOF_DATES = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}


def asof_date(year: int, tag: str) -> pd.Timestamp:
    month, day = ASOF_DATES[tag]
    return pd.Timestamp(year=int(year), month=month, day=day)


def load_geometries() -> dict[int, dict]:
    fields = pd.read_csv(DATA_RAW / "cropwise_fields_full.csv", usecols=["id", "shape_simplified_geojson"])
    fields = fields.dropna(subset=["shape_simplified_geojson"]).copy()
    return {int(r.id): json.loads(r.shape_simplified_geojson) for r in fields.itertuples(index=False)}


def load_tasks() -> pd.DataFrame:
    # Canonical row list = v26 harvest-truth (our current target set), with a
    # fallback to v24 if v26 is not built yet. Using v26 ensures every modeled
    # field-year gets an S2 lookup (v24 missed ~172 harvest field-years).
    v26 = DATA_PROCESSED / "ml_dataset_v26_harvest_asof_08_01.csv"
    src = v26 if v26.exists() else DATA_PROCESSED / "ml_dataset_v24_agro_asof_08_01.csv"
    base = pd.read_csv(src, usecols=["field_id", "year", "standard_name", "target_yield_t_ha"])
    base = base[base["target_yield_t_ha"] >= 1.0].copy()
    rows = []
    for tag in ASOF_DATES:
        tmp = base.copy()
        tmp["asof_tag"] = tag
        rows.append(tmp)
    tasks = pd.concat(rows, ignore_index=True)
    return tasks.drop_duplicates(["field_id", "year", "asof_tag"]).reset_index(drop=True)


def done_keys() -> set[tuple[int, int, str]]:
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV, usecols=["field_id", "year", "asof_tag", "status"])
    df = df[df["status"].eq("ok")].copy()
    return set(map(tuple, df[["field_id", "year", "asof_tag"]].astype({"field_id": int, "year": int}).values.tolist()))


def field_bbox(geom: dict) -> list[float]:
    return list(shape(geom).bounds)


def read_band(href: str, geom_wgs84: dict) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        geom_src = shapely_transform(transformer.transform, shape(geom_wgs84))
        arr, _ = mask(src, [mapping(geom_src)], crop=True, filled=False)
        data = arr[0].astype("float32").filled(np.nan)
        if np.nanmax(data) > 2:
            data = data / 10000.0
        valid = np.isfinite(data) & (data > 0)
        return data, valid


def robust_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.nanmean(x)) if len(x) else np.nan


def compute_indices(item, geom: dict) -> dict[str, object]:
    signed = planetary_computer.sign(item)

    def href(key: str) -> str:
        return signed.assets[key].href

    b02, m02 = read_band(href("B02"), geom)
    b03, m03 = read_band(href("B03"), geom)
    b04, m04 = read_band(href("B04"), geom)
    b08, m08 = read_band(href("B08"), geom)
    valid10 = m02 & m03 & m04 & m08

    ndvi = (b08 - b04) / (b08 + b04)
    evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0)
    gcvi = b08 / b03 - 1.0

    b05, m05 = read_band(href("B05"), geom)
    b8a, m8a = read_band(href("B8A"), geom)
    valid20 = m05 & m8a
    ndre = (b8a - b05) / (b8a + b05)

    return {
        "s2_item_id": item.id,
        "s2_datetime": item.datetime.isoformat() if item.datetime else None,
        "s2_cloud_cover": item.properties.get("eo:cloud_cover"),
        "s2_pixels_10m": int(valid10.sum()),
        "s2_pixels_20m": int(valid20.sum()),
        "s2_ndvi_mean": robust_mean(np.where(valid10, ndvi, np.nan)),
        "s2_ndre_mean": robust_mean(np.where(valid20, ndre, np.nan)),
        "s2_evi_mean": robust_mean(np.where(valid10, evi, np.nan)),
        "s2_gcvi_mean": robust_mean(np.where(valid10, gcvi, np.nan)),
    }


def find_item(catalog: Client, geom: dict, year: int, tag: str, cloud_lt: float):
    end = asof_date(year, tag)
    start = end - pd.Timedelta(days=30)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=field_bbox(geom),
        datetime=f"{start.date().isoformat()}/{end.date().isoformat()}",
        query={"eo:cloud_cover": {"lt": cloud_lt}},
        max_items=20,
    )
    items = list(search.items())
    if not items:
        return None
    items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 1000), abs((it.datetime.replace(tzinfo=None) - end).days)))
    return items[0]


def write_row(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process N new tasks.")
    parser.add_argument("--cloud-lt", type=float, default=35.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    geometries = load_geometries()
    tasks = load_tasks()
    done = done_keys()
    catalog = Client.open(STAC_URL)

    fieldnames = [
        "field_id",
        "year",
        "standard_name",
        "asof_tag",
        "s2_item_id",
        "s2_datetime",
        "s2_cloud_cover",
        "s2_pixels_10m",
        "s2_pixels_20m",
        "s2_ndvi_mean",
        "s2_ndre_mean",
        "s2_evi_mean",
        "s2_gcvi_mean",
        "status",
        "error_message",
    ]
    header = not OUT_CSV.exists()
    n_new = 0
    with OUT_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if header:
            writer.writeheader()

        for row in tasks.itertuples(index=False):
            key = (int(row.field_id), int(row.year), str(row.asof_tag))
            if key in done:
                continue
            geom = geometries.get(int(row.field_id))
            if geom is None:
                out = {"field_id": row.field_id, "year": row.year, "standard_name": row.standard_name, "asof_tag": row.asof_tag, "status": "missing_geometry"}
                writer.writerow(out)
                continue
            try:
                item = find_item(catalog, geom, int(row.year), str(row.asof_tag), args.cloud_lt)
                if item is None:
                    out = {"field_id": row.field_id, "year": row.year, "standard_name": row.standard_name, "asof_tag": row.asof_tag, "status": "no_scene"}
                else:
                    out = {
                        "field_id": int(row.field_id),
                        "year": int(row.year),
                        "standard_name": row.standard_name,
                        "asof_tag": row.asof_tag,
                        **compute_indices(item, geom),
                        "status": "ok",
                    }
            except Exception as e:
                out = {
                    "field_id": int(row.field_id),
                    "year": int(row.year),
                    "standard_name": row.standard_name,
                    "asof_tag": row.asof_tag,
                    "status": f"error:{type(e).__name__}",
                    "error_message": str(e)[:200],
                }
            writer.writerow(out)
            fh.flush()
            n_new += 1
            if n_new % 10 == 0:
                print(f"processed new={n_new}; last={key}; status={out.get('status')}")
            if args.limit is not None and n_new >= args.limit:
                break
            time.sleep(args.sleep)
    print(f"done new={n_new}; output={OUT_CSV}")


if __name__ == "__main__":
    main()
