"""
Smoke test for Sentinel-2 field-level indices via Microsoft Planetary Computer.

Goal: prove that we can compute indices missing from Cropwise API:
  - NDRE = (B08 - B05) / (B08 + B05)
  - EVI  = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
  - GCVI = B08 / B03 - 1

This script does ONE field / ONE date window. If it works, use the batch script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_RAW = Path("data_raw")
OUT = Path("reports/microscope")
OUT.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def load_field(field_id: int | None = None) -> tuple[int, dict]:
    fields = pd.read_csv(DATA_RAW / "cropwise_fields_full.csv")
    fields = fields[fields["shape_simplified_geojson"].notna()].copy()
    if field_id is None:
        # A known field with complete target/satellite coverage.
        field_id = int(fields.iloc[0]["id"])
    row = fields[fields["id"].astype(int) == int(field_id)].iloc[0]
    geom = json.loads(row["shape_simplified_geojson"])
    return int(field_id), geom


def field_bbox(geom: dict) -> list[float]:
    shp = shape(geom)
    return list(shp.bounds)


def signed_asset_href(item, asset_key: str) -> str:
    signed = planetary_computer.sign(item)
    return signed.assets[asset_key].href


def read_band(href: str, geom_wgs84: dict, *, scale_reflectance: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        geom_src = shapely_transform(transformer.transform, shape(geom_wgs84))
        arr, _ = mask(src, [mapping(geom_src)], crop=True, filled=False)
        masked = arr[0]
        data = masked.astype("float32").filled(np.nan)
        if scale_reflectance and src.scales and src.scales[0] not in (None, 1.0):
            data = data * src.scales[0]
        # Sentinel-2 L2A reflectance assets are scaled by 10000 in PC.
        if scale_reflectance and np.nanmax(data) > 2:
            data = data / 10000.0
        valid = np.isfinite(data) & (data > 0)
        return data, valid


def robust_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.nanmean(x))


def compute_indices(item, geom: dict) -> dict:
    hrefs = {
        "B02": signed_asset_href(item, "B02"),
        "B03": signed_asset_href(item, "B03"),
        "B04": signed_asset_href(item, "B04"),
        "B05": signed_asset_href(item, "B05"),
        "B08": signed_asset_href(item, "B08"),
        "B8A": signed_asset_href(item, "B8A"),
        "SCL": signed_asset_href(item, "SCL"),
    }
    b02, m02 = read_band(hrefs["B02"], geom)
    b03, m03 = read_band(hrefs["B03"], geom)
    b04, m04 = read_band(hrefs["B04"], geom)
    b08, m08 = read_band(hrefs["B08"], geom)
    # 10m indices: B02/B03/B04/B08 share shape.
    valid = m02 & m03 & m04 & m08
    evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0)
    gcvi = b08 / b03 - 1.0
    ndvi = (b08 - b04) / (b08 + b04)

    # 20m red-edge index: B05 and B8A share shape.
    b05, m05 = read_band(hrefs["B05"], geom)
    b8a, m8a = read_band(hrefs["B8A"], geom)
    valid_re = m05 & m8a
    ndre = (b8a - b05) / (b8a + b05)

    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
        "pixels": int(valid.sum()),
        "ndvi_mean": robust_mean(np.where(valid, ndvi, np.nan)),
        "ndre_mean": robust_mean(np.where(valid_re, ndre, np.nan)),
        "evi_mean": robust_mean(np.where(valid, evi, np.nan)),
        "gcvi_mean": robust_mean(np.where(valid, gcvi, np.nan)),
    }


def main() -> None:
    field_id, geom = load_field()
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=field_bbox(geom),
        datetime="2023-07-01/2023-07-20",
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=10,
    )
    items = list(search.items())
    print(f"field_id={field_id}; found items={len(items)}")
    if not items:
        return
    # Pick least-cloudy item.
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 1000))
    result = compute_indices(items[0], geom)
    result["field_id"] = field_id
    out = pd.DataFrame([result])
    out_path = OUT / "sentinel2_indices_smoke.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
