"""
v24 dataset: v23 + agronomic fixes/additions driven by literature review.

Changes vs v23:
  1. FIX: soil_moisture parsing. The API stores soil_moisture as
     [date, [L0, L1, L2, L3]] (ERA5-Land depth layers 0-7/7-28/28-100/100-289 cm).
     v23's scalar parser silently dropped all of it -> cw_sm_* were 100% NaN.
     Here we extract root-zone moisture = mean(L0, L1) per day, then aggregate.
  2. ADD: growing-degree-days (GDD) accumulation to as-of from the daily
     temperature series, with crop-aware base temperature. Phenology alignment
     is repeatedly shown to improve yield models.
  3. ADD: heat stress = number of days with Tmax-proxy >= 30C to as-of.
  4. DROP dead columns from the feature space happens in the trainer
     (region_id/district_id/till_type are ~100% missing).

Literature basis (see reports/V24_AGRO_AUDIT_RU.md):
  - sunflower yield in N. Kazakhstan steppe is water/phosphorus limited;
    soil moisture + temperature + flowering-window greenness matter most;
  - wheat yield is best explained around anthesis (peak greenness) plus
    max temperature during grain fill.

Outputs:
  data_processed/ml_dataset_v24_agro_asof_{07_01,08_01,09_01}.csv
  reports/V24_DATASET_SUMMARY.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys

sys.path.insert(0, str(ROOT))

from scripts.asof.build_v23_cropwise_full_dataset import (  # noqa: E402
    ASOF_DATES,
    add_soil_asof,
    load_base,
    load_soil_long,
    series_features,
    sowing_features,
)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")

# Crop base temperature for GDD (deg C). Wheat/cool-season ~0-5, sunflower ~6-8.
GDD_BASE = {
    "wheat_spring": 0.0,
    "wheat_winter": 0.0,
    "barley_spring": 0.0,
    "rye_winter": 0.0,
    "avena_spring": 0.0,
    "pea": 4.0,
    "lentil": 4.0,
    "soya": 8.0,
    "maize": 8.0,
    "sunflower": 6.0,
    "oil_seed_raps_spring": 5.0,
    "oil_seed_raps_winter": 5.0,
}
DEFAULT_GDD_BASE = 5.0
HEAT_THRESHOLD_C = 30.0


def parse_scalar_series(value: object) -> list[tuple[pd.Timestamp, float]]:
    if pd.isna(value):
        return []
    try:
        raw = json.loads(str(value))
    except Exception:
        return []
    out = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        date = pd.to_datetime(item[0], errors="coerce")
        try:
            val = float(item[1])
        except (TypeError, ValueError):
            continue
        if pd.notna(date) and np.isfinite(val):
            out.append((pd.Timestamp(date), val))
    out.sort(key=lambda x: x[0])
    return out


def parse_soil_moisture_series(value: object) -> list[tuple[pd.Timestamp, float]]:
    """soil_moisture value is [date, [L0,L1,L2,L3]]; use root-zone = mean(L0,L1)."""
    if pd.isna(value):
        return []
    try:
        raw = json.loads(str(value))
    except Exception:
        return []
    out = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        date = pd.to_datetime(item[0], errors="coerce")
        layers = item[1]
        if not isinstance(layers, list) or not layers:
            continue
        root = [float(x) for x in layers[:2] if isinstance(x, (int, float))]
        if pd.isna(date) or not root:
            continue
        out.append((pd.Timestamp(date), float(np.mean(root))))
    out.sort(key=lambda x: x[0])
    return out


def load_satellite(base: pd.DataFrame) -> dict[tuple[int, int, str], list[tuple[pd.Timestamp, float]]]:
    needed = set(map(tuple, base[["field_id", "year"]].astype(int).drop_duplicates().values.tolist()))
    sat = pd.read_csv(DATA_RAW / "cropwise_historical_values_full.csv")
    sat = sat[sat["product_type"].isin(["ndvi", "temperature", "soil_moisture"])].copy()
    sat["field_id"] = sat["field_id"].astype(int)
    sat["year"] = sat["year"].astype(int)
    sat = sat[[key in needed for key in zip(sat["field_id"], sat["year"])]].copy()
    print(f"Satellite rows to parse: {len(sat):,} for {len(needed):,} target field-years")
    mapping = {}
    for i, row in enumerate(sat.itertuples(index=False), start=1):
        product = str(row.product_type)
        if product == "soil_moisture":
            parsed = parse_soil_moisture_series(row.value_json)
        else:
            parsed = parse_scalar_series(row.value_json)
        mapping[(int(row.field_id), int(row.year), product)] = parsed
        if i % 1000 == 0:
            print(f"  parsed satellite rows: {i:,}/{len(sat):,}")
    return mapping


def temperature_agro_features(
    values: list[tuple[pd.Timestamp, float]],
    year: int,
    month: int,
    day: int,
    base_temp: float,
) -> dict[str, float]:
    asof = pd.Timestamp(year=int(year), month=month, day=day)
    vals = [(d, v) for d, v in values if d.year == int(year) and d <= asof]
    if not vals:
        return {
            "gdd_to_asof": np.nan,
            "heat_days_ge30_to_asof": np.nan,
            "cold_days_le0_to_asof": np.nan,
        }
    temps = np.array([v for _, v in vals], dtype=float)
    gdd = float(np.sum(np.clip(temps - base_temp, 0.0, None)))
    return {
        "gdd_to_asof": gdd,
        "heat_days_ge30_to_asof": int(np.sum(temps >= HEAT_THRESHOLD_C)),
        "cold_days_le0_to_asof": int(np.sum(temps <= 0.0)),
    }


def build_one(base: pd.DataFrame, sat: dict, soil: pd.DataFrame, tag: str, month: int, day: int) -> pd.DataFrame:
    rows = []
    for row in base.itertuples(index=False):
        rec = row._asdict()
        rec["asof_tag"] = tag
        rec.update(sowing_features(rec.get("sowing_date"), int(rec["year"]), month, day))

        ndvi = sat.get((int(rec["field_id"]), int(rec["year"]), "ndvi"), [])
        temp = sat.get((int(rec["field_id"]), int(rec["year"]), "temperature"), [])
        sm = sat.get((int(rec["field_id"]), int(rec["year"]), "soil_moisture"), [])

        rec.update(series_features(ndvi, int(rec["year"]), month, day, "ndvi"))
        rec.update(series_features(temp, int(rec["year"]), month, day, "cw_temp"))
        rec.update(series_features(sm, int(rec["year"]), month, day, "cw_sm"))

        base_temp = GDD_BASE.get(str(rec.get("standard_name")), DEFAULT_GDD_BASE)
        rec.update(temperature_agro_features(temp, int(rec["year"]), month, day, base_temp))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = add_soil_asof(out, soil, month, day)
    out = out[out["ndvi_n_obs_asof"] >= 3].reset_index(drop=True)
    return out


def main() -> None:
    base = load_base()
    print(f"Base target rows: {len(base):,}, fields={base['field_id'].nunique():,}")
    sat = load_satellite(base)
    soil = load_soil_long()

    summary = []
    for tag, (month, day) in ASOF_DATES.items():
        out = build_one(base, sat, soil, tag, month, day)
        path = DATA_PROCESSED / f"ml_dataset_v24_agro_asof_{tag}.csv"
        out.to_csv(path, index=False)
        summary.append(
            {
                "asof_tag": tag,
                "rows": len(out),
                "fields": out["field_id"].nunique(),
                "sm_coverage": f"{out['cw_sm_mean_asof'].notna().mean():.0%}",
                "gdd_coverage": f"{out['gdd_to_asof'].notna().mean():.0%}",
                "soil_coverage": f"{out['field_soil_pH'].notna().mean():.0%}",
            }
        )
        print(
            f"{tag}: rows={len(out):,}, cols={len(out.columns)}, "
            f"sm_cov={out['cw_sm_mean_asof'].notna().mean():.0%} -> {path}"
        )

    sm = pd.DataFrame(summary)
    md = ["# v24 agronomic dataset summary\n\n"]
    md.append("v23 + soil_moisture fix (root-zone) + GDD + heat/cold stress.\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V24_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
