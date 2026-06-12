"""
Sprint 1.2 — Build clean ML dataset on top of the cleaned target.

Strategy:
  1. Take `data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv` as feature base
     (it has NDVI seasonal + phase + soil + weather features, but a dirty target).
  2. Replace target_yield_t_ha with the cleaned values from `targets_cleaned_t_ha.csv`.
     Inner join on (field_id, year). Records with no clean target are dropped.
  3. Drop features:
     - field_lat, field_long, field_group_id   (1 farm, almost-constant)
     - field_soil_CEC, field_soil_Ca_saturation, field_soil_N_NO3   (>=99% NaN)
     - harvest leakage: ops_count_harvesting, ops_days_seeding_to_harvest,
       ops_season_end, ops_season_duration_days, ops_yield_t_ha
       (these encode post-harvest information — leakage for in-season forecast)
  4. Keep the rest, save to `data_processed/ml_dataset_clean_v2.csv`.
  5. Save audit JSON: row counts, dropped features, target source breakdown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

BASE_DATASET = DATA_PROCESSED / "ml_dataset_full_extended_cropwindow_v1_phases.csv"
CLEAN_TARGET = DATA_PROCESSED / "targets_cleaned_t_ha.csv"
OUT_DATASET = DATA_PROCESSED / "ml_dataset_clean_v2.csv"
AUDIT_JSON = DATA_PROCESSED / "ml_dataset_clean_v2_audit.json"

DROP_FEATURES = [
    # Geography on a single farm — constants
    "field_lat",
    "field_long",
    "field_group_id",
    # Almost entirely missing
    "field_soil_CEC",
    "field_soil_Ca_saturation",
    "field_soil_N_NO3",
    # Harvest-time leakage (post-harvest info)
    "ops_count_harvesting",
    "ops_days_seeding_to_harvest",
    "ops_season_end",
    "ops_season_duration_days",
    "ops_yield_t_ha",
    "ops_season_start",   # date string, not numeric; would need encoding anyway
]


def main():
    print("=" * 80)
    print("BUILD CLEAN ML DATASET v2")
    print("=" * 80)

    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"Missing base dataset: {BASE_DATASET}")
    if not CLEAN_TARGET.exists():
        raise FileNotFoundError(f"Missing clean target: {CLEAN_TARGET}. Run build_clean_targets.py first.")

    base = pd.read_csv(BASE_DATASET)
    print(f"Base dataset: {len(base)} rows, {len(base.columns)} cols")

    target = pd.read_csv(CLEAN_TARGET)
    target = target[target["kept"]].copy()
    print(f"Clean target: {len(target)} kept rows (across {target['std_name'].nunique()} crop types)")

    # Replace target. Inner join — drop rows with no cleaned target.
    base = base.drop(columns=["target_yield_t_ha"], errors="ignore")
    base["field_id"] = base["field_id"].astype(int)
    base["year"] = base["year"].astype(int)
    target["field_id"] = target["field_id"].astype(int)
    target["year"] = target["year"].astype(int)

    target_slim = target[["field_id", "year", "target_yield_t_ha", "target_source", "unit_fix_applied"]]
    merged = base.merge(target_slim, on=["field_id", "year"], how="inner")
    print(f"After target replace + inner join: {len(merged)} rows")

    # Drop trash features (only those that exist)
    dropped_existing = [c for c in DROP_FEATURES if c in merged.columns]
    merged = merged.drop(columns=dropped_existing)
    print(f"Dropped {len(dropped_existing)} features: {dropped_existing}")

    # Reorder so target is at the front (after IDs)
    front = ["field_id", "year", "crop_id", "prev_crop_id", "target_yield_t_ha", "target_source", "unit_fix_applied"]
    front = [c for c in front if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    merged.to_csv(OUT_DATASET, index=False)
    print(f"\nSaved {OUT_DATASET}")
    print(f"Final shape: {merged.shape}")

    # Audit
    audit = {
        "base_rows": int(len(base)),
        "clean_target_rows": int(len(target)),
        "merged_rows": int(len(merged)),
        "dropped_features": dropped_existing,
        "kept_features": [c for c in merged.columns if c not in front],
        "target_source_breakdown": merged["target_source"].value_counts().to_dict(),
        "year_distribution": merged["year"].value_counts().sort_index().to_dict(),
        "crop_distribution": merged["crop_id"].value_counts().to_dict(),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"Saved audit: {AUDIT_JSON}")

    # Console summary of target distribution per crop
    crops = pd.read_csv("data_raw/crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    summary = merged.merge(crops, on="crop_id", how="left")
    print("\nTarget per crop (cleaned):")
    print(
        summary.groupby("standard_name")
        .agg(
            n=("target_yield_t_ha", "size"),
            t_min=("target_yield_t_ha", "min"),
            t_med=("target_yield_t_ha", "median"),
            t_max=("target_yield_t_ha", "max"),
        )
        .sort_values("n", ascending=False)
        .to_string()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
