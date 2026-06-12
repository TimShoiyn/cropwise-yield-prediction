"""
Full Cropwise pull with the refreshed token.

Endpoints confirmed working (2026-06):
  - GET /api/v3/fields            -> geometry (shape_simplified_geojson) + meta
  - GET /api/v3/soil_tests        -> soil element panels
  - GET /api/v3a/historical_values -> historical series

Pagination is cursor-based via `from_id` + `limit` (max 100); meta.response
carries first_record_id / last_record_id.

Satellite series are stored COMPACTLY: one row per (field_id, year, product_type)
with the daily [date, value] list as a JSON string (same shape as
productivity_estimate_peers.ndvi_values). This keeps the file small and is
directly parseable by the existing as-of feature builders.

Token: read from env CROPWISE_API_KEY (do not hardcode).

Usage:
  CROPWISE_API_KEY=... python3 scripts/external/fetch_cropwise_full.py fields
  CROPWISE_API_KEY=... python3 scripts/external/fetch_cropwise_full.py soil
  CROPWISE_API_KEY=... python3 scripts/external/fetch_cropwise_full.py satellite
  CROPWISE_API_KEY=... python3 scripts/external/fetch_cropwise_full.py all

Outputs (data_raw/):
  cropwise_fields_full.csv
  cropwise_soil_tests_full.csv
  cropwise_historical_values_full.csv      (resumable; appended per field/product)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("data_raw")
OUT.mkdir(exist_ok=True)

BASE = "https://operations.cropwise.com"
TOKEN = os.getenv("CROPWISE_API_KEY", "")
SLEEP = float(os.getenv("CROPWISE_SLEEP", "0.15"))
SAT_PRODUCTS_DEFAULT = ["ndvi", "temperature", "soil_moisture"]

FIELDS_CSV = OUT / "cropwise_fields_full.csv"
SOIL_CSV = OUT / "cropwise_soil_tests_full.csv"
SAT_CSV = OUT / "cropwise_historical_values_full.csv"


def session() -> requests.Session:
    if not TOKEN:
        raise SystemExit("Set CROPWISE_API_KEY env var with the API token.")
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "X-User-Api-Token": TOKEN})
    return s


def get_json(s: requests.Session, path: str, params: dict, retries: int = 4) -> dict:
    url = BASE + path
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url} params={params}")


def paginate(s: requests.Session, path: str, extra: dict | None = None):
    from_id = 0
    while True:
        params = {"from_id": from_id, "limit": 100}
        if extra:
            params.update(extra)
        d = get_json(s, path, params)
        rows = d.get("data", [])
        if not rows:
            break
        yield rows
        last = d.get("meta", {}).get("response", {}).get("last_record_id")
        if len(rows) < 100 or last is None or last == from_id:
            break
        from_id = last


def fetch_fields(s: requests.Session) -> pd.DataFrame:
    all_rows = []
    for rows in paginate(s, "/api/v3/fields"):
        all_rows.extend(rows)
    df = pd.json_normalize(all_rows).drop_duplicates(subset=["id"])
    df.to_csv(FIELDS_CSV, index=False)
    print(f"fields: {len(df)} -> {FIELDS_CSV}")
    return df


def fetch_soil(s: requests.Session) -> pd.DataFrame:
    all_rows = []
    for rows in paginate(s, "/api/v3/soil_tests"):
        all_rows.extend(rows)
    df = pd.json_normalize(all_rows)
    df.to_csv(SOIL_CSV, index=False)
    print(f"soil_tests: {len(df)} -> {SOIL_CSV}")
    return df


# Extra paginated collections fetched with the refreshed token.
COLLECTIONS = {
    "crops": ("/api/v3/crops", OUT / "cropwise_crops_full.csv"),
    "agro_operations": ("/api/v3/agro_operations", OUT / "cropwise_agro_operations_full.csv"),
    "productivity_estimates": ("/api/v3/productivity_estimates", OUT / "cropwise_productivity_estimates_full.csv"),
    "productivity_estimate_histories": (
        "/api/v3/productivity_estimate_histories",
        OUT / "cropwise_productivity_estimate_histories_full.csv",
    ),
    "yield_maps": ("/api/v3/yield_maps", OUT / "cropwise_yield_maps_full.csv"),
    "seasons": ("/api/v3/seasons", OUT / "cropwise_seasons_full.csv"),
    "field_shapes": ("/api/v3/field_shapes", OUT / "cropwise_field_shapes_full.csv"),
    "fertilizers": ("/api/v3/fertilizers", OUT / "cropwise_fertilizers_full.csv"),
    "chemicals": ("/api/v3/chemicals", OUT / "cropwise_chemicals_full.csv"),
    "plant_threats": ("/api/v3/plant_threats", OUT / "cropwise_plant_threats_full.csv"),
    "agri_work_plans": ("/api/v3/agri_work_plans", OUT / "cropwise_agri_work_plans_full.csv"),
}


def fetch_collection(s: requests.Session, name: str) -> pd.DataFrame:
    path, out_csv = COLLECTIONS[name]
    all_rows = []
    page = 0
    for rows in paginate(s, path):
        all_rows.extend(rows)
        page += 1
        if page % 20 == 0:
            print(f"  {name}: {len(all_rows)} rows so far...")
    df = pd.json_normalize(all_rows)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    df.to_csv(out_csv, index=False)
    print(f"{name}: {len(df)} -> {out_csv}")
    return df


def _field_ids() -> list[int]:
    if not FIELDS_CSV.exists():
        raise SystemExit("Run 'fields' first to produce cropwise_fields_full.csv")
    return sorted(pd.read_csv(FIELDS_CSV)["id"].dropna().astype(int).unique().tolist())


def _done_fields() -> set:
    if not SAT_CSV.exists():
        return set()
    df = pd.read_csv(SAT_CSV, usecols=["field_id"])
    return set(df["field_id"].dropna().astype(int).unique().tolist())


def fetch_satellite(s: requests.Session, products: list[str]) -> None:
    # NOTE: the historical_values `type` param is ignored by the API; a single
    # request per field returns ALL product_types it has (ndvi, temperature,
    # soil_moisture, ndvi_s2a/s2b/s2c/ps8/l8). So we query once per field.
    field_ids = _field_ids()
    done = _done_fields()
    header_needed = not SAT_CSV.exists()
    total = len(field_ids)
    done_n = 0
    with SAT_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if header_needed:
            writer.writerow(["field_id", "year", "product_type", "n_obs", "value_json"])
        for fid in field_ids:
            done_n += 1
            if fid in done:
                continue
            try:
                d = get_json(s, "/api/v3a/historical_values", {"field_id": fid})
            except RuntimeError as e:
                print(f"  skip field={fid}: {e}")
                continue
            for rec in d.get("data", []):
                series = rec.get("value") or []
                writer.writerow([
                    fid,
                    rec.get("year"),
                    rec.get("product_type"),
                    len(series),
                    json.dumps(series, separators=(",", ":")),
                ])
            fh.flush()
            if done_n % 50 == 0:
                print(f"  progress {done_n}/{total} (field={fid})")
            time.sleep(SLEEP)
    print(f"satellite done -> {SAT_CSV}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    s = session()
    if cmd in ("fields", "all"):
        fetch_fields(s)
    if cmd in ("soil", "all"):
        fetch_soil(s)
    if cmd in COLLECTIONS:
        fetch_collection(s, cmd)
    if cmd == "collections":
        for name in COLLECTIONS:
            fetch_collection(s, name)
    if cmd == "all":
        for name in COLLECTIONS:
            fetch_collection(s, name)
    if cmd in ("satellite", "all"):
        products = SAT_PRODUCTS_DEFAULT
        if cmd == "satellite" and len(sys.argv) > 2:
            products = [p.strip() for p in sys.argv[2].split(",") if p.strip()]
        fetch_satellite(s, products)


if __name__ == "__main__":
    main()
