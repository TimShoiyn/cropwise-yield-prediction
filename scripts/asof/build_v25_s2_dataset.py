"""
v25 dataset builder: v24 agronomic dataset + Sentinel-2 red-edge/EVI/GCVI.

Inputs:
  data_processed/ml_dataset_v24_agro_asof_{tag}.csv  (per as-of)
  data_raw/sentinel2_indices_b1.csv                  (B1 output, status==ok)

Merge key: (field_id, year, asof_tag).

Sentinel-2 columns added (single best scene in 30d before as-of):
  s2_ndvi_mean, s2_ndre_mean, s2_evi_mean, s2_gcvi_mean,
  s2_cloud_cover, s2_pixels_10m, s2_pixels_20m.

We also derive simple cross-index features the literature highlights:
  s2_ndre_minus_ndvi  (red-edge information beyond NDVI saturation)
  s2_evi_minus_ndvi

Outputs:
  data_processed/ml_dataset_v25_s2_asof_{tag}.csv
  reports/V25_DATASET_SUMMARY.md

Run AFTER B1 extraction finishes (or partially: rows without S2 keep NaN).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
REPORTS = Path("reports")

ASOF_TAGS = ["07_01", "08_01", "09_01"]
S2_PATH = DATA_RAW / "sentinel2_indices_b1.csv"

S2_VALUE_COLS = [
    "s2_ndvi_mean",
    "s2_ndre_mean",
    "s2_evi_mean",
    "s2_gcvi_mean",
    "s2_cloud_cover",
    "s2_pixels_10m",
    "s2_pixels_20m",
]


def load_s2() -> pd.DataFrame:
    if not S2_PATH.exists():
        raise SystemExit(f"Missing {S2_PATH}. Run B1 extractor first.")
    s2 = pd.read_csv(S2_PATH)
    s2 = s2[s2["status"] == "ok"].copy()
    s2["field_id"] = s2["field_id"].astype(int)
    s2["year"] = s2["year"].astype(int)
    s2 = s2.drop_duplicates(["field_id", "year", "asof_tag"], keep="last")
    keep = ["field_id", "year", "asof_tag"] + [c for c in S2_VALUE_COLS if c in s2.columns]
    return s2[keep]


def build_one(tag: str, s2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v24_agro_asof_{tag}.csv")
    df["field_id"] = df["field_id"].astype(int)
    df["year"] = df["year"].astype(int)

    s2_tag = s2[s2["asof_tag"] == tag].drop(columns=["asof_tag"])
    merged = df.merge(s2_tag, on=["field_id", "year"], how="left")

    # Cross-index features (only where S2 present).
    if "s2_ndre_mean" in merged and "s2_ndvi_mean" in merged:
        merged["s2_ndre_minus_ndvi"] = merged["s2_ndre_mean"] - merged["s2_ndvi_mean"]
    if "s2_evi_mean" in merged and "s2_ndvi_mean" in merged:
        merged["s2_evi_minus_ndvi"] = merged["s2_evi_mean"] - merged["s2_ndvi_mean"]

    info = {
        "asof_tag": tag,
        "rows": len(merged),
        "rows_with_s2": int(merged["s2_ndre_mean"].notna().sum()) if "s2_ndre_mean" in merged else 0,
        "s2_coverage": f"{merged['s2_ndre_mean'].notna().mean():.0%}" if "s2_ndre_mean" in merged else "0%",
    }
    return merged, info


def main() -> None:
    s2 = load_s2()
    print(f"S2 ok rows: {len(s2):,}; field-year-asof keys: {s2[['field_id','year']].drop_duplicates().shape[0]:,}")

    summary = []
    for tag in ASOF_TAGS:
        merged, info = build_one(tag, s2)
        path = DATA_PROCESSED / f"ml_dataset_v25_s2_asof_{tag}.csv"
        merged.to_csv(path, index=False)
        summary.append(info)
        print(f"{tag}: rows={info['rows']:,}, with_s2={info['rows_with_s2']:,} ({info['s2_coverage']}) -> {path}")

    sm = pd.DataFrame(summary)
    md = ["# v25 dataset summary (v24 + Sentinel-2 indices)\n\n"]
    md.append("Sentinel-2 NDRE/EVI/GCVI merged from B1 (`data_raw/sentinel2_indices_b1.csv`).\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n\nNote: rows without a usable S2 scene keep NaN; CatBoost handles missing values.\n")
    (REPORTS / "V25_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
