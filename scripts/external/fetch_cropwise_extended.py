"""
Cropwise Operations API extended fetcher.

Goal: re-download fields, ndvi_timeseries, operations, soil_tests and
field_scout_reports for ALL field_ids that we already see in
history_items_full.csv (currently 633 fields, vs. 30 covered).

Usage:
    CROPWISE_API_TOKEN=<your_token> python3 scripts/external/fetch_cropwise_extended.py --check
        # quickly probes a single endpoint to validate token

    CROPWISE_API_TOKEN=<your_token> python3 scripts/external/fetch_cropwise_extended.py --fetch fields ndvi
        # full pull of selected resources

Notes:
- Endpoints: https://operations.cropwise.com/api/v3
- Auth: header X-User-Api-Token
- Rate limit: ~5 rps; we sleep 0.25s between calls and retry on 429/5xx.
- Data is appended into data_raw_v2/ to avoid clobbering original CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import urllib.request
import urllib.parse
import urllib.error

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

API_BASE = "https://operations.cropwise.com/api/v3"
RAW_OUT = Path("data_raw_v2")
RAW_OUT.mkdir(parents=True, exist_ok=True)

REQUEST_SLEEP_S = 0.25
MAX_RETRIES = 5
TIMEOUT_S = 30


class ApiError(RuntimeError):
    pass


def _http_get(url: str, token: str) -> Any:
    req = urllib.request.Request(url, headers={
        "X-User-Api-Token": token,
        "Accept": "application/json",
        "User-Agent": "phd-research/1.0",
    })
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                wait = REQUEST_SLEEP_S * (2 ** attempt)
                print(f"  [warn] {e.code} on {url} retry {attempt}/{MAX_RETRIES} in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            if e.code == 401:
                raise ApiError("401 Unauthorized — token is missing/invalid/expired") from e
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            wait = REQUEST_SLEEP_S * (2 ** attempt)
            print(f"  [warn] {e} on {url} retry {attempt}/{MAX_RETRIES} in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
            last_err = e
    raise ApiError(f"max retries exceeded for {url}: {last_err}")


def _paged(path: str, token: str, params: dict | None = None) -> Iterable[dict]:
    """Iterate all pages of a Cropwise list endpoint."""
    page = 1
    base_params = dict(params or {})
    base_params.setdefault("per_page", 200)
    while True:
        q = dict(base_params)
        q["page"] = page
        url = f"{API_BASE}{path}?{urllib.parse.urlencode(q)}"
        payload = _http_get(url, token)
        if not payload:
            break
        if isinstance(payload, dict) and "data" in payload:
            items = payload["data"]
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        if not items:
            break
        for it in items:
            yield it
        if isinstance(payload, dict):
            meta = payload.get("meta") or {}
            total_pages = meta.get("total_pages")
            if total_pages and page >= int(total_pages):
                break
        if len(items) < base_params["per_page"]:
            break
        page += 1
        time.sleep(REQUEST_SLEEP_S)


def cmd_check(token: str) -> int:
    print(f"Probing {API_BASE}/fields?per_page=1 ...")
    try:
        items = list(_paged("/fields", token, {"per_page": 1}))
        print(f"OK — got {len(items)} field record(s).")
        if items:
            sample = items[0]
            print("Sample keys:", list(sample.keys())[:20])
            f_id = sample.get("id")
            print(f"Sample field id: {f_id}")
        return 0
    except ApiError as e:
        print(f"FAIL: {e}")
        return 2


def cmd_fields(token: str) -> None:
    out = RAW_OUT / "fields_v2.csv"
    print(f"Downloading all fields -> {out}")
    rows = list(_paged("/fields", token))
    print(f"  got {len(rows)} fields")
    pd.DataFrame(rows).to_csv(out, index=False)


def cmd_history_items(token: str) -> None:
    out = RAW_OUT / "history_items_v2.csv"
    print(f"Downloading all history_items -> {out}")
    rows = list(_paged("/history_items", token))
    print(f"  got {len(rows)} history_items")
    pd.DataFrame(rows).to_csv(out, index=False)


def cmd_ndvi(token: str, field_ids: list[int]) -> None:
    out = RAW_OUT / "ndvi_timeseries_v2.csv"
    print(f"Downloading ndvi_timeseries for {len(field_ids)} fields -> {out}")
    chunks: list[pd.DataFrame] = []
    for i, fid in enumerate(field_ids, 1):
        try:
            rows = list(_paged(f"/fields/{fid}/ndvi_timeseries", token))
            if rows:
                df = pd.DataFrame(rows)
                df["field_id"] = fid
                chunks.append(df)
            if i % 20 == 0:
                print(f"  [{i}/{len(field_ids)}] field {fid}: {len(rows)} obs (running total {sum(len(c) for c in chunks)})")
        except ApiError as e:
            print(f"  [error] field {fid}: {e}")
        time.sleep(REQUEST_SLEEP_S)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_csv(out, index=False)
        print(f"  wrote {out}")


def cmd_operations(token: str) -> None:
    out = RAW_OUT / "operations_v2.csv"
    print(f"Downloading all operations -> {out}")
    rows = list(_paged("/operations", token))
    print(f"  got {len(rows)} operations")
    pd.DataFrame(rows).to_csv(out, index=False)


def cmd_soil_tests(token: str) -> None:
    out = RAW_OUT / "soil_tests_v2.csv"
    print(f"Downloading all soil_tests -> {out}")
    rows = list(_paged("/soil_tests", token))
    print(f"  got {len(rows)} soil_tests")
    pd.DataFrame(rows).to_csv(out, index=False)


def cmd_scout(token: str) -> None:
    out = RAW_OUT / "field_scout_reports_v2.csv"
    print(f"Downloading all field_scout_reports -> {out}")
    rows = list(_paged("/field_scout_reports", token))
    print(f"  got {len(rows)} reports")
    pd.DataFrame(rows).to_csv(out, index=False)


def list_target_field_ids() -> list[int]:
    hi = pd.read_csv("data_raw/history_items_full.csv", low_memory=False, usecols=["field_id"])
    return sorted(int(x) for x in hi["field_id"].dropna().unique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="probe a single endpoint")
    parser.add_argument("--fetch", nargs="*",
                        choices=["fields", "history_items", "ndvi", "operations", "soil_tests", "scout", "all"],
                        default=[],
                        help="resources to download")
    parser.add_argument("--token", default=os.environ.get("CROPWISE_API_TOKEN", ""))
    parser.add_argument("--fields-from", default="history",
                        help="how to choose field_ids for ndvi: 'history' (from history_items_full)")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: CROPWISE_API_TOKEN not set (or use --token).", file=sys.stderr)
        sys.exit(1)

    if args.check:
        sys.exit(cmd_check(args.token))

    if not args.fetch:
        print("Nothing to do. Pass --check or --fetch <res>.")
        sys.exit(0)

    res = set(args.fetch)
    if "all" in res:
        res = {"fields", "history_items", "ndvi", "operations", "soil_tests", "scout"}

    if "fields" in res:
        cmd_fields(args.token)
    if "history_items" in res:
        cmd_history_items(args.token)
    if "ndvi" in res:
        ids = list_target_field_ids()
        print(f"Will pull NDVI for {len(ids)} fields")
        cmd_ndvi(args.token, ids)
    if "operations" in res:
        cmd_operations(args.token)
    if "soil_tests" in res:
        cmd_soil_tests(args.token)
    if "scout" in res:
        cmd_scout(args.token)


if __name__ == "__main__":
    main()
