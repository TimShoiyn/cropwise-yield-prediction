"""Clean & normalize the extra Cropwise collections into canonical files.

Inputs (data_raw/):
  - cropwise_agro_operations_full.csv
  - cropwise_productivity_estimates_full.csv
  - cropwise_yield_maps_full.csv
  - cropwise_crops_full.csv

Outputs (data_clean/):
  - harvest_yield_clean.csv        ground-truth yield from harvest ops (t/ha)
  - cropwise_estimates_clean.csv   Cropwise productivity estimates (benchmark only)
  - yield_maps_clean.csv           combine yield maps avg (t/ha)
  - crops_lookup.csv               crop id -> name lookup
  - operations_clean.csv           all management ops, normalized dates/areas

Run:
  python3 scripts/external/clean_cropwise_collections.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data_raw")
OUT = Path("data_clean")
OUT.mkdir(exist_ok=True)

# Realistic per-ha yield bounds for the region (t/ha). Anything outside is unit noise.
YIELD_MIN = 0.1
YIELD_MAX = 16.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.year


def clean_crops() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cropwise_crops_full.csv", low_memory=False)
    keep = [c for c in ["id", "name", "short_name", "standard_name", "season_type", "multi_year"] if c in df.columns]
    out = df[keep].drop_duplicates(subset=["id"]).sort_values("id")
    out.to_csv(OUT / "crops_lookup.csv", index=False)
    print(f"crops_lookup: {len(out)} -> {OUT/'crops_lookup.csv'}")
    return out


def clean_harvest() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cropwise_agro_operations_full.csv", low_memory=False)
    harv = df[df.operation_type.astype(str).str.contains("harv", case=False, na=False)].copy()

    harv["harvested_weight"] = _num(harv["harvested_weight"])
    harv["completed_area"] = _num(harv["completed_area"])
    harv["year"] = _num(harv["season"]).astype("Int64")

    # Yield = harvested tonnes / completed hectares.
    harv["yield_t_ha"] = harv["harvested_weight"] / harv["completed_area"].replace(0, np.nan)

    valid = harv[(harv["completed_area"] > 0) & harv["yield_t_ha"].between(YIELD_MIN, YIELD_MAX)].copy()

    cols = {
        "field_id": "field_id",
        "year": "year",
        "yield_t_ha": "yield_t_ha",
        "harvested_weight": "harvested_weight_t",
        "completed_area": "completed_area_ha",
        "completed_date": "completed_date",
        "humidity": "humidity",
        "protein_content": "protein_content",
        "oil_content": "oil_content",
        "id": "operation_id",
    }
    cols = {k: v for k, v in cols.items() if k in valid.columns}
    out = valid[list(cols)].rename(columns=cols)

    # One harvest record per (field, year): largest completed area wins (main harvest).
    out = out.sort_values("completed_area_ha", ascending=False).drop_duplicates(subset=["field_id", "year"])
    out = out.sort_values(["field_id", "year"]).reset_index(drop=True)
    out.to_csv(OUT / "harvest_yield_clean.csv", index=False)
    print(
        f"harvest_yield: {len(out)} rows / {out.field_id.nunique()} fields / "
        f"years {int(out.year.min())}-{int(out.year.max())} -> {OUT/'harvest_yield_clean.csv'}"
    )
    return out


def clean_estimates() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cropwise_productivity_estimates_full.csv", low_memory=False)
    df["estimate_value"] = _num(df["estimate_value"])
    # API stores estimate_value in centner/ha (ц/га); convert to t/ha to match harvest.
    df["cropwise_t_ha"] = df["estimate_value"] / 10.0
    keep = [c for c in ["field_id", "year", "estimate_value", "cropwise_t_ha", "estimate_date", "history_item_id"] if c in df.columns]
    out = df[keep].dropna(subset=["estimate_value"])
    out = out.sort_values(["field_id", "year"]).drop_duplicates(subset=["field_id", "year"], keep="last")
    out = out.reset_index(drop=True)
    out.to_csv(OUT / "cropwise_estimates_clean.csv", index=False)
    print(
        f"cropwise_estimates: {len(out)} rows / {out.field_id.nunique()} fields -> {OUT/'cropwise_estimates_clean.csv'}"
    )
    return out


def clean_yield_maps() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cropwise_yield_maps_full.csv", low_memory=False)
    y = df[(df.property_name == "yield") & (df.units == "tonn_per_ha")].copy()
    y["calculated_average"] = _num(y["calculated_average"])
    y["external_average"] = _num(y.get("external_average"))
    y["year"] = _year(y["created_at"]).astype("Int64")
    y = y[y["calculated_average"].between(YIELD_MIN, YIELD_MAX)]
    keep = ["field_id", "year", "calculated_average", "external_average", "id"]
    keep = [c for c in keep if c in y.columns]
    out = y[keep].rename(columns={"calculated_average": "map_yield_t_ha", "id": "yield_map_id"})
    out = out.sort_values("map_yield_t_ha", ascending=False).drop_duplicates(subset=["field_id", "year"])
    out = out.sort_values(["field_id", "year"]).reset_index(drop=True)
    out.to_csv(OUT / "yield_maps_clean.csv", index=False)
    print(f"yield_maps: {len(out)} rows / {out.field_id.nunique()} fields -> {OUT/'yield_maps_clean.csv'}")
    return out


def clean_operations() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cropwise_agro_operations_full.csv", low_memory=False)
    df["year"] = _num(df["season"]).astype("Int64")
    keep = [
        "id", "field_id", "year", "operation_type", "operation_subtype",
        "status", "planned_area", "completed_area", "completed_date",
        "planned_start_date", "actual_start_datetime",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].sort_values(["field_id", "year", "operation_type"]).reset_index(drop=True)
    out.to_csv(OUT / "operations_clean.csv", index=False)
    print(f"operations: {len(out)} rows / {out.field_id.nunique()} fields -> {OUT/'operations_clean.csv'}")
    return out


def main() -> None:
    clean_crops()
    h = clean_harvest()
    e = clean_estimates()
    m = clean_yield_maps()
    clean_operations()

    # Cross-source coverage: how many harvest fields also have a Cropwise estimate?
    overlap = pd.merge(h[["field_id", "year"]], e[["field_id", "year"]], on=["field_id", "year"])
    print(
        f"\nharvest∩cropwise_estimate (same field+year): {len(overlap)} rows "
        f"-> usable for an honest benchmark"
    )
    overlap_m = pd.merge(h[["field_id", "year"]], m[["field_id", "year"]], on=["field_id", "year"])
    print(f"harvest∩yield_map (same field+year): {len(overlap_m)} rows")


if __name__ == "__main__":
    main()
