"""Microscope audit of the v26 / B2 / B2.1 data pipeline.

Reproduces every integrity check we ran by hand, so the thesis has a single
runnable artifact. Prints a numbered report; writes nothing.

Run:
  python3 scripts/audit/audit_v26_microscope.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

RAW = Path("data_raw")
CLEAN = Path("data_clean")
PROC = Path("data_processed")
ASOF_MD = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}


def asof_series(year: pd.Series, month: int, day: int) -> np.ndarray:
    return pd.to_datetime(dict(year=year, month=month, day=day)).values


def main() -> None:
    print("=" * 70)
    print("v26 MICROSCOPE AUDIT")
    print("=" * 70)

    ao = pd.read_csv(RAW / "cropwise_agro_operations_full.csv", low_memory=False)
    harv = ao[ao.operation_type.astype(str).str.contains("harv", case=False, na=False)].copy()
    harv["ca"] = pd.to_numeric(harv.completed_area, errors="coerce")
    harv["pa"] = pd.to_numeric(harv.planned_area, errors="coerce")

    g = harv.groupby(["field_id", "season"]).size()
    print(f"\n[1] harvest ops: {len(harv)}; field-years with >1 op: {(g > 1).sum()} (max {g.max()})")
    sub = harv[(harv.ca > 0) & (harv.pa > 0)]
    ratio = sub.ca / sub.pa
    print(f"    completed/planned area: median={ratio.median():.2f}, 5th pct={ratio.quantile(.05):.2f}")

    h = pd.read_csv(CLEAN / "harvest_yield_clean.csv")
    h["cd"] = pd.to_datetime(h.completed_date, errors="coerce", utc=True).dt.tz_localize(None)
    print("\n[2] harvest timing vs as-of (post-harvest = invalid forecast):")
    for tag, (mo, d) in ASOF_MD.items():
        asof = asof_series(h.year, mo, d)
        before = (h.cd.values < asof).sum()
        print(f"    {tag}: already harvested before as-of = {before}/{len(h)} ({before/len(h)*100:.0f}%)")

    hi = pd.read_csv(RAW / "history_items_full.csv", low_memory=False).dropna(subset=["field_id", "year", "crop_id"])
    hi["field_id"] = hi.field_id.astype(int)
    hi["year"] = hi.year.astype(int)
    hi["crop_id"] = hi.crop_id.astype(int)
    amb = hi.groupby(["field_id", "year"]).crop_id.nunique()
    print(f"\n[3] crop label ambiguity: field-years with >1 crop_id = {(amb > 1).sum()}/{len(amb)}")

    e = pd.read_csv(CLEAN / "cropwise_estimates_clean.csv")[["field_id", "year", "estimate_value"]]
    cr = pd.read_csv(CLEAN / "crops_lookup.csv")[["id", "standard_name"]]
    hi2 = hi.merge(cr, left_on="crop_id", right_on="id", how="left").drop_duplicates(["field_id", "year"])
    m = h.merge(hi2[["field_id", "year", "standard_name"]], on=["field_id", "year"], how="left").merge(
        e, on=["field_id", "year"], how="left"
    )
    m = m[(m.yield_t_ha > 0.5) & (m.estimate_value > 0)]
    m["ratio"] = m.estimate_value / m.yield_t_ha
    print("\n[4] Cropwise/harvest ratio per crop (~10 => centner/ha, /10 correct):")
    print(m.groupby("standard_name").ratio.median().round(2).sort_values().to_string())

    print("\n[5] as-of Cropwise date integrity (history_date must be <= as-of):")
    b = pd.read_csv(CLEAN / "cropwise_estimates_asof_clean.csv")
    b["hd"] = pd.to_datetime(b.cropwise_history_date, errors="coerce", utc=True).dt.tz_localize(None)
    for tag, (mo, d) in ASOF_MD.items():
        s = b[b.asof_tag == tag]
        asof = asof_series(s.year, mo, d)
        print(f"    {tag}: violations={int((s.hd.values > asof).sum())}; values>12 t/ha={int((s.cropwise_asof_t_ha > 12).sum())}")

    print("\n[6] built dataset sanity (08_01):")
    ds = pd.read_csv(PROC / "ml_dataset_v26_harvest_asof_08_01.csv")
    print(f"    rows={len(ds)}, dup (field,year)={ds.duplicated(['field_id', 'year']).sum()}")
    print(
        f"    ndvi_mean [{ds.ndvi_mean_asof.min():.2f},{ds.ndvi_mean_asof.max():.2f}]; "
        f"cw_sm_mean [{ds.cw_sm_mean_asof.min():.2f},{ds.cw_sm_mean_asof.max():.2f}]"
    )
    print("\nDONE")


if __name__ == "__main__":
    main()
