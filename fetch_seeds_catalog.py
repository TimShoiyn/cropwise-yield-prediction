"""
Fetch Seeds catalog (seed_id -> crop_id, name, etc.) from Cropwise Operations API.

Why:
  In operations.csv, application_mix_items contains Seed items with applicable_id == seed_id,
  but does NOT include crop_id / crop name. Endpoint /api/v3/seeds/{id} returns crop_id.

Input:
  - data_raw/operations.csv (extract unique seed IDs from application_mix_items)

Output:
  - data_raw/seeds.csv (full seed objects, one row per seed_id)
  - data_processed/seed_id_to_crop_id.csv (minimal mapping)

Run (PowerShell):
  $env:CROPWISE_API_KEY="..."; python fetch_seeds_catalog.py
"""

from __future__ import annotations

import ast
import os
import time
from typing import Any

import pandas as pd
import requests


OPS_CSV = "data_raw/operations.csv"
OUT_SEEDS_CSV = "data_raw/seeds.csv"
OUT_MAPPING_CSV = "data_processed/seed_id_to_crop_id.csv"

BASE_URL = "https://operations.cropwise.com/api/v3"
TOKEN = os.getenv("CROPWISE_API_KEY", "")
TIMEOUT_SEC = 20
SLEEP_SEC = 0.25


def _headers(token: str) -> dict[str, str]:
    return {"X-User-Api-Token": token, "Accept": "application/json"}


def _safe_parse_mix_items(raw: object) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s or s.lower() == "nan":
        return []
    try:
        items = ast.literal_eval(s)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return []
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]
    return []


def _extract_seed_ids(ops_df: pd.DataFrame) -> list[int]:
    if "application_mix_items" not in ops_df.columns:
        return []
    seed_ids: set[int] = set()
    non_null = ops_df[ops_df["application_mix_items"].notna()]
    for _, row in non_null.iterrows():
        items = _safe_parse_mix_items(row["application_mix_items"])
        for it in items:
            if it.get("applicable_type") == "Seed" and it.get("applicable_id") is not None:
                try:
                    seed_ids.add(int(it["applicable_id"]))
                except Exception:
                    continue
    return sorted(seed_ids)


def fetch_seed(seed_id: int, token: str) -> dict[str, Any] | None:
    url = f"{BASE_URL}/seeds/{seed_id}"
    r = requests.get(url, headers=_headers(token), timeout=TIMEOUT_SEC)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    # API returns {"data": {...}}
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data


def main() -> None:
    if not TOKEN:
        raise SystemExit("❌ CROPWISE_API_KEY not set. Set env var and re-run.")

    print("📂 Loading operations:", OPS_CSV)
    ops_df = pd.read_csv(OPS_CSV)
    seed_ids = _extract_seed_ids(ops_df)
    print(f"🌱 Unique seed_ids in operations mix_items: {len(seed_ids)}")
    if not seed_ids:
        raise SystemExit("❌ No seed_ids found in operations.csv application_mix_items.")

    records: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []

    for i, sid in enumerate(seed_ids, start=1):
        try:
            obj = fetch_seed(sid, TOKEN)
            if obj is None:
                print(f"  [{i}/{len(seed_ids)}] seed_id={sid}: 404 Not Found")
                continue
            records.append(obj)
            crop_id = obj.get("crop_id") if isinstance(obj, dict) else None
            name = obj.get("name") if isinstance(obj, dict) else None
            print(f"  [{i}/{len(seed_ids)}] seed_id={sid}: crop_id={crop_id}, name={name}")
        except Exception as e:
            errors.append((sid, str(e)))
            print(f"  [{i}/{len(seed_ids)}] seed_id={sid}: ERROR {e}")
        time.sleep(SLEEP_SEC)

    if not records:
        raise SystemExit("❌ Failed to fetch any seeds.")

    seeds_df = pd.json_normalize(records)
    seeds_df.to_csv(OUT_SEEDS_CSV, index=False)
    print(f"💾 Saved: {OUT_SEEDS_CSV} ({len(seeds_df)} rows)")

    mapping = seeds_df[["id", "crop_id", "name"]].copy()
    mapping.rename(columns={"id": "seed_id", "name": "seed_name"}, inplace=True)
    mapping.to_csv(OUT_MAPPING_CSV, index=False)
    print(f"💾 Saved: {OUT_MAPPING_CSV} ({len(mapping)} rows)")

    if errors:
        print("\n⚠️ Errors (seed_id -> error):")
        for sid, msg in errors[:20]:
            print(f"  - {sid}: {msg}")
        if len(errors) > 20:
            print(f"  ... {len(errors)-20} more")


if __name__ == "__main__":
    main()

