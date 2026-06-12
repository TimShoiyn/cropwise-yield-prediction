"""
Critical question: why is our v7 dataset only 30 fields / 201 rows?

soil_tests covers 421 fields. history_items has thousands of records.
But ndvi_timeseries.csv is only 30 fields. Find the bottleneck.
"""

from __future__ import annotations
import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")

print("=" * 70)
print("Field-id cardinality across raw and processed sources")
print("=" * 70)

def n_fields(path: Path, col: str = "field_id") -> int:
    if not path.exists():
        return -1
    try:
        df = pd.read_csv(path, usecols=[col])
        return df[col].nunique()
    except Exception as e:
        return -2

print(f"fields.csv .id                    : {pd.read_csv(DATA_RAW/'fields.csv', usecols=['id'])['id'].nunique()}")
print(f"history_items_full.csv field_id   : {n_fields(DATA_RAW/'history_items_full.csv')}")
print(f"ndvi_timeseries.csv field_id      : {n_fields(DATA_RAW/'ndvi_timeseries.csv')}")
print(f"openmeteo_daily.csv field_id      : {n_fields(DATA_RAW/'openmeteo_daily.csv')}")
print(f"soil_tests.csv field_id           : {n_fields(DATA_RAW/'soil_tests.csv')}")
print(f"operations.csv field_id           : {n_fields(DATA_RAW/'operations.csv')}")
print(f"yield_maps.csv field_id           : {n_fields(DATA_RAW/'yield_maps.csv')}")
print(f"field_scout_reports field_id      : {n_fields(DATA_RAW/'field_scout_reports_aggregated.csv')}")
print(f"plant_threats (catalog)           : N/A (no field_id - this is master)")

print()
print(f"targets_factual_t_ha.csv field_id : {n_fields(DATA_PROCESSED/'targets_factual_t_ha.csv')}")
print(f"targets_factual kept=True only    : ", end="")
tf = pd.read_csv(DATA_PROCESSED/'targets_factual_t_ha.csv')
print(tf[tf["kept"]]["field_id"].nunique())
print(f"  rows kept                      : {tf['kept'].sum()}")
print(f"  rows total                     : {len(tf)}")

print()
print("=" * 70)
print("history_items broken down by data presence")
print("=" * 70)
hi = pd.read_csv(DATA_RAW/'history_items_full.csv', low_memory=False,
                 usecols=["field_id", "year", "crop_id", "productivity", "harvested_weight"])
print(f"Total rows: {len(hi)}")
print(f"Has productivity        : {hi['productivity'].notna().sum()}")
print(f"Has harvested_weight    : {hi['harvested_weight'].notna().sum()}")
print(f"Has either              : {(hi['productivity'].notna() | hi['harvested_weight'].notna()).sum()}")
print(f"Distinct fields w/ either: {hi[hi['productivity'].notna() | hi['harvested_weight'].notna()]['field_id'].nunique()}")
print(f"Year range with data     : {hi[hi['productivity'].notna()|hi['harvested_weight'].notna()]['year'].min()}..{hi[hi['productivity'].notna()|hi['harvested_weight'].notna()]['year'].max()}")

# Per-year row counts (where target exists)
hi_with = hi[hi['productivity'].notna() | hi['harvested_weight'].notna()]
print()
print("Rows per year with any factual target:")
print(hi_with.groupby("year").size().sort_index().to_string())

print()
print("=" * 70)
print("Why ndvi_timeseries only 30 fields?")
print("=" * 70)
ndvi = pd.read_csv(DATA_RAW/'ndvi_timeseries.csv', usecols=["field_id"])
fields_in_ndvi = set(ndvi['field_id'].unique())
fields_with_target = set(hi_with['field_id'].unique())
overlap = fields_in_ndvi & fields_with_target
print(f"NDVI fields: {len(fields_in_ndvi)}")
print(f"Fields with target in history: {len(fields_with_target)}")
print(f"Overlap: {len(overlap)}")
print(f"Fields with target but NO NDVI: {len(fields_with_target - fields_in_ndvi)}")
print(f"Fields with NDVI but NO target: {len(fields_in_ndvi - fields_with_target)}")
print()
print("First 20 fields with target but no NDVI:")
print(sorted(fields_with_target - fields_in_ndvi)[:20])
print()
print("First 20 NDVI fields:")
print(sorted(fields_in_ndvi)[:20])
print()
print("Field range in ndvi:", min(fields_in_ndvi), "..", max(fields_in_ndvi))
print("Field range with target:", min(fields_with_target), "..", max(fields_with_target))

# Could be: ndvi is only for a subset of "main" fields. Check fields.csv
print()
fcsv = pd.read_csv(DATA_RAW/'fields.csv')
print("Field IDs in fields.csv:", sorted(fcsv['id'].tolist()))
