"""v26 dataset (Roadmap B3): train on REAL harvest ground-truth, 528 fields.

Big change vs v23/v24: the target is no longer history_items productivity, it is
the direct combine harvest: harvested_weight (t) / completed_area (ha), pulled from
agro_operations and cleaned in data_clean/harvest_yield_clean.csv (528 fields,
1499 field-years, 2021-2025).

Crop label comes from history_items (field_id, year -> crop_id) + crops_lookup.
Cropwise benchmark comes from data_clean/cropwise_estimates_clean.csv (ц/га -> t/ha).

All as-of features (NDVI / temperature / soil_moisture / GDD / heat-cold stress /
soil panels / sowing / rotation) are reused verbatim from the v23/v24 builders, so
the only thing that changes is the (much larger, much more honest) target.

Outputs:
  data_processed/ml_dataset_v26_harvest_asof_{07_01,08_01,09_01}.csv
  reports/V26_DATASET_SUMMARY.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.asof.build_v23_cropwise_full_dataset import (  # noqa: E402
    ASOF_DATES,
    MAX_T_HA,
    DEFAULT_MAX_T_HA,
    add_rotation_features,
    add_soil_asof,
    load_soil_long,
    series_features,
    sowing_features,
)
from scripts.asof.build_v24_agronomic_dataset import (  # noqa: E402
    GDD_BASE,
    DEFAULT_GDD_BASE,
    load_satellite,
    temperature_agro_features,
)

DATA_RAW = Path("data_raw")
DATA_CLEAN = Path("data_clean")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")


def load_harvest_base() -> pd.DataFrame:
    harvest = pd.read_csv(DATA_CLEAN / "harvest_yield_clean.csv")
    harvest["field_id"] = harvest["field_id"].astype(int)
    harvest["year"] = harvest["year"].astype(int)

    # Crop label + sowing/harvest dates + variety/till from history_items (dedup per field-year).
    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False)
    hi["field_id"] = pd.to_numeric(hi["field_id"], errors="coerce")
    hi["year"] = pd.to_numeric(hi["year"], errors="coerce")
    hi = hi.dropna(subset=["field_id", "year", "crop_id"]).copy()
    hi["field_id"] = hi["field_id"].astype(int)
    hi["year"] = hi["year"].astype(int)
    hi["crop_id"] = hi["crop_id"].astype(int)
    # Prefer the 'active' record, then the most recently updated, as the field's crop.
    sort_cols = [c for c in ["active", "updated_at"] if c in hi.columns]
    if sort_cols:
        hi = hi.sort_values(sort_cols, ascending=False)
    hi = hi.drop_duplicates(subset=["field_id", "year"], keep="first")
    hi_keep = [c for c in ["field_id", "year", "crop_id", "variety", "till_type",
                           "sowing_date", "harvesting_date"] if c in hi.columns]
    hi = hi[hi_keep]

    crops = pd.read_csv(DATA_CLEAN / "crops_lookup.csv")[["id", "standard_name"]].rename(
        columns={"id": "crop_id"}
    )

    fields = pd.read_csv(DATA_RAW / "cropwise_fields_full.csv").rename(
        columns={"id": "field_id", "name": "field_name"}
    )
    field_cols = [c for c in ["field_id", "field_name", "tillable_area", "calculated_area",
                              "lat", "long"] if c in fields.columns]
    fields = fields[field_cols].copy()

    est = pd.read_csv(DATA_CLEAN / "cropwise_estimates_clean.csv")[["field_id", "year", "cropwise_t_ha"]]
    est["field_id"] = est["field_id"].astype(int)
    est["year"] = est["year"].astype(int)

    base = (
        harvest.merge(hi, on=["field_id", "year"], how="left")
        .merge(crops, on="crop_id", how="left")
        .merge(fields, on="field_id", how="left")
        .merge(est, on=["field_id", "year"], how="left")
    )

    base = base.dropna(subset=["standard_name"]).copy()
    base["target_yield_t_ha"] = base["yield_t_ha"].astype(float)
    base["target_source"] = "v26_harvest_op"

    # Per-crop sanity cap (drop physically impossible harvest rows for the crop).
    caps = base["standard_name"].map(lambda c: MAX_T_HA.get(str(c), DEFAULT_MAX_T_HA))
    base = base[base["target_yield_t_ha"] <= caps].copy()

    base = base.rename(
        columns={"tillable_area": "field_tillable_area", "calculated_area": "field_calculated_area"}
    )
    return add_rotation_features(base)


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
    base = load_harvest_base()
    print(
        f"Harvest base: {len(base):,} rows, fields={base['field_id'].nunique():,}, "
        f"years={int(base['year'].min())}-{int(base['year'].max())}, "
        f"cropwise_cov={base['cropwise_t_ha'].notna().mean():.0%}"
    )
    sat = load_satellite(base)
    soil = load_soil_long()

    summary = []
    for tag, (month, day) in ASOF_DATES.items():
        out = build_one(base, sat, soil, tag, month, day)
        path = DATA_PROCESSED / f"ml_dataset_v26_harvest_asof_{tag}.csv"
        out.to_csv(path, index=False)
        summary.append(
            {
                "asof_tag": tag,
                "rows": len(out),
                "fields": out["field_id"].nunique(),
                "cropwise_pairs": int(out["cropwise_t_ha"].notna().sum()),
                "sm_cov": f"{out['cw_sm_mean_asof'].notna().mean():.0%}",
                "soil_cov": f"{out['field_soil_pH'].notna().mean():.0%}",
                "target_mean": round(float(out["target_yield_t_ha"].mean()), 2),
            }
        )
        print(f"{tag}: rows={len(out):,}, fields={out['field_id'].nunique():,}, cols={len(out.columns)} -> {path}")

    sm = pd.DataFrame(summary)
    md = ["# v26 harvest-truth dataset summary (Roadmap B3)\n\n"]
    md.append("Target = real combine harvest (harvested_weight / completed_area), 528 fields.\n")
    md.append("Crop from history_items, Cropwise benchmark from cropwise_estimates_clean.\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V26_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
