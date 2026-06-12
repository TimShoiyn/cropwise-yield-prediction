"""
Build factual yield targets only.

This intentionally excludes:
  - productivity_estimates.csv
  - history_items_full.productivity_estimate
  - expected_yield

Allowed target sources:
  1. harvested_weight / tillable_area
  2. history_items_full.productivity after per-crop unit normalization

The output also preserves sowing_date / harvesting_date because v9 uses real
sowing dates for GDD- and NDVI-aligned features.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MAX_T_HA: dict[str, float] = {
    "sunflower": 5.0,
    "wheat_spring": 7.0,
    "wheat_winter": 8.0,
    "barley_spring": 7.0,
    "barley_winter": 8.0,
    "maize": 12.0,
    "oil_seed_raps_spring": 4.5,
    "oil_seed_raps_winter": 5.0,
    "soya": 4.0,
    "pea": 4.5,
    "lentil": 3.5,
    "buckwheat": 3.0,
    "safflower": 2.5,
}
DEFAULT_MAX_T_HA = 5.0

MIN_T_HA: dict[str, float] = {
    "sunflower": 0.3,
    "wheat_spring": 0.3,
    "wheat_winter": 0.3,
    "barley_spring": 0.3,
    "barley_winter": 0.3,
    "maize": 0.5,
    "oil_seed_raps_spring": 0.2,
    "oil_seed_raps_winter": 0.2,
    "soya": 0.2,
    "pea": 0.2,
}
DEFAULT_MIN_T_HA = 0.2


def normalize_unit(raw_value: float, std_name: str | None) -> tuple[float, bool]:
    if not pd.notna(raw_value):
        return (np.nan, False)
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if raw_value > cap:
        return (float(raw_value) / 10.0, True)
    return (float(raw_value), False)


def is_in_realistic_range(t_ha: float, std_name: str | None) -> bool:
    if not pd.notna(t_ha):
        return False
    lo = MIN_T_HA.get(std_name or "", DEFAULT_MIN_T_HA)
    hi = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA) * 1.2
    return lo <= t_ha <= hi


def build() -> pd.DataFrame:
    print("=" * 80)
    print("BUILD FACTUAL TARGETS ONLY")
    print("=" * 80)

    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "tillable_area"]].rename(columns={"id": "field_id"})
    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False)

    keep_cols = [
        "field_id",
        "year",
        "crop_id",
        "productivity",
        "harvested_weight",
        "sowing_date",
        "harvesting_date",
    ]
    hi = hi[keep_cols].copy()
    hi = hi.merge(crops, on="crop_id", how="left")
    hi = hi.merge(fields, on="field_id", how="left")

    hi["field_id"] = pd.to_numeric(hi["field_id"], errors="coerce")
    hi["year"] = pd.to_numeric(hi["year"], errors="coerce")
    hi["sowing_date"] = pd.to_datetime(hi["sowing_date"], errors="coerce")
    hi["harvesting_date"] = pd.to_datetime(hi["harvesting_date"], errors="coerce")

    hi["physical_t_ha"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(
        hi["tillable_area"], errors="coerce"
    )

    prod = pd.to_numeric(hi["productivity"], errors="coerce")
    norm = [normalize_unit(v, s) for v, s in zip(prod, hi["standard_name"])]
    hi["productivity_t_ha"] = [x[0] for x in norm]
    hi["productivity_unit_fixed"] = [x[1] for x in norm]

    rows = []
    keys = (
        hi.dropna(subset=["field_id", "year"])
        .sort_values(["field_id", "year", "sowing_date"])
        .drop_duplicates(["field_id", "year"], keep="last")
    )

    for _, row in keys.iterrows():
        std = row.get("standard_name")
        target = np.nan
        source = "no_data"
        unit_fixed = False

        phys = row.get("physical_t_ha")
        if pd.notna(phys) and is_in_realistic_range(float(phys), std):
            target = float(phys)
            source = "physical_harvested_weight"
        else:
            prod_t = row.get("productivity_t_ha")
            if pd.notna(prod_t) and is_in_realistic_range(float(prod_t), std):
                target = float(prod_t)
                source = "productivity_normalized"
                unit_fixed = bool(row.get("productivity_unit_fixed", False))

        rows.append(
            {
                "field_id": int(row["field_id"]),
                "year": int(row["year"]),
                "crop_id": int(row["crop_id"]) if pd.notna(row.get("crop_id")) else np.nan,
                "standard_name": std,
                "target_yield_t_ha": target,
                "target_source": source,
                "unit_fix_applied": unit_fixed,
                "sowing_date": row.get("sowing_date"),
                "harvesting_date": row.get("harvesting_date"),
                "kept": pd.notna(target),
            }
        )

    out = pd.DataFrame(rows).sort_values(["field_id", "year"]).reset_index(drop=True)
    out_path = DATA_PROCESSED / "targets_factual_t_ha.csv"
    out.to_csv(out_path, index=False)

    kept = out[out["kept"]].copy()
    audit = (
        kept.groupby(["standard_name", "target_source"])
        .agg(
            n=("target_yield_t_ha", "size"),
            min_t_ha=("target_yield_t_ha", "min"),
            median_t_ha=("target_yield_t_ha", "median"),
            max_t_ha=("target_yield_t_ha", "max"),
            unit_fixed_n=("unit_fix_applied", "sum"),
            sowing_date_coverage=("sowing_date", lambda s: s.notna().mean()),
        )
        .reset_index()
    )
    audit_path = DATA_PROCESSED / "targets_factual_audit.csv"
    audit.to_csv(audit_path, index=False)

    md = ["# Factual Target Audit\n\n"]
    md.append("Generated by `scripts/clean/build_factual_targets.py`.\n\n")
    md.append("Excluded Cropwise estimates from target construction.\n\n")
    md.append(f"- total keys: {len(out)}\n")
    md.append(f"- kept factual targets: {int(out['kept'].sum())}\n")
    md.append(f"- dropped/no factual target: {int((~out['kept']).sum())}\n\n")
    md.append("## By crop/source\n\n")
    md.append(audit.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "factual_target_audit.md").write_text("".join(md), encoding="utf-8")

    print(f"Saved {out_path}: {len(out)} rows, kept={int(out['kept'].sum())}")
    print(f"Saved {audit_path}")
    print(f"Saved {REPORTS / 'factual_target_audit.md'}")
    return out


if __name__ == "__main__":
    build()
