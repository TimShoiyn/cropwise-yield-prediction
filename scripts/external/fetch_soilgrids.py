"""
Sprint 6.1 — Fetch SoilGrids 250m for all fields.

API: https://rest.isric.org/soilgrids/v2.0/properties/query
  - free, no API key, public
  - rate limit: ~5 req/minute for anonymous (we use 15s sleep)

Properties (all standard SoilGrids variables):
  clay, sand, silt        (texture, %)
  soc, nitrogen           (organic carbon, total N, dg/kg)
  bdod                    (bulk density, kg/dm³)
  cec                     (cation exchange capacity, mmol(c)/kg)
  phh2o                   (pH in water)

Depths used:
  0-5cm, 5-15cm, 15-30cm  (rooting zone for annual crops)

Output: data_raw/soilgrids.csv (long → wide format, one row per field).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["clay", "sand", "silt", "soc", "nitrogen", "bdod", "cec", "phh2o"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

REQUEST_SLEEP_S = 8.0
GRID_RESOLUTION = 0.02    # ~2km. Farm radius ~5km, so this is enough.


def fetch_one(lat: float, lon: float) -> dict | None:
    params = [("lon", f"{lon:.5f}"), ("lat", f"{lat:.5f}"), ("value", "mean")]
    for p in PROPERTIES:
        params.append(("property", p))
    for d in DEPTHS:
        params.append(("depth", d))
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    error: {e}")
        return None


def parse_response(data: dict, lat: float, lon: float) -> dict:
    out = {"lat_grid": lat, "lon_grid": lon}
    if not data or "properties" not in data:
        return out
    layers = data["properties"].get("layers", [])
    for layer in layers:
        prop = layer.get("name")
        depths = layer.get("depths", [])
        for d in depths:
            depth_label = d.get("label", "")
            mean = d.get("values", {}).get("mean")
            if mean is None:
                continue
            unit = layer.get("unit_measure", {})
            d_factor = unit.get("d_factor", 1.0) or 1.0
            real_value = mean / d_factor
            key = f"sg_{prop}_{depth_label.replace('cm', '').replace('-', '_')}"
            out[key] = float(real_value)
    return out


def main():
    print("=" * 80)
    print("FETCH SOILGRIDS 250m")
    print("=" * 80)
    fields = pd.read_csv("data_raw/fields.csv")[["id", "lat", "long"]].rename(columns={"id": "field_id"})
    fields = fields.dropna(subset=["lat", "long"]).copy()
    fields["field_id"] = fields["field_id"].astype(int)
    fields["lat"] = pd.to_numeric(fields["lat"], errors="coerce")
    fields["long"] = pd.to_numeric(fields["long"], errors="coerce")
    fields = fields.dropna(subset=["lat", "long"])
    fields["lat_grid"] = (fields["lat"] / GRID_RESOLUTION).round() * GRID_RESOLUTION
    fields["lon_grid"] = (fields["long"] / GRID_RESOLUTION).round() * GRID_RESOLUTION
    cells = fields.drop_duplicates(subset=["lat_grid", "lon_grid"])[["lat_grid", "lon_grid"]].reset_index(drop=True)
    print(f"  {len(fields)} fields → {len(cells)} unique soil cells (resolution {GRID_RESOLUTION}°)")
    print(f"  properties: {PROPERTIES}")
    print(f"  depths: {DEPTHS}")
    print()

    cell_data: dict[tuple[float, float], dict] = {}
    for i, row in cells.iterrows():
        lat = float(row["lat_grid"]); lon = float(row["lon_grid"])
        attempts = 0
        rec = None
        while attempts < 3:
            data = fetch_one(lat, lon)
            if data is not None:
                rec = parse_response(data, lat, lon)
                break
            attempts += 1
            time.sleep(REQUEST_SLEEP_S * attempts)
        if rec is None:
            print(f"  [{i+1:2d}/{len(cells)}] cell ({lat:.4f}, {lon:.4f}) GAVE UP")
            continue
        cell_data[(lat, lon)] = rec
        print(f"  [{i+1:2d}/{len(cells)}] cell ({lat:.4f}, {lon:.4f}) ok ({len(rec)-2} props)")
        time.sleep(REQUEST_SLEEP_S)

    if not cell_data:
        print("Nothing fetched."); return

    rows = []
    for _, row in fields.iterrows():
        fid = int(row["field_id"])
        key = (float(row["lat_grid"]), float(row["lon_grid"]))
        if key not in cell_data:
            continue
        rec = dict(cell_data[key])
        rec["field_id"] = fid
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv("data_raw/soilgrids.csv", index=False)
    print(f"\nSaved data_raw/soilgrids.csv: {len(out)} rows × {out.shape[1]} cols")
    print(out.describe(include='all').T.head(30).to_string())


if __name__ == "__main__":
    main()
