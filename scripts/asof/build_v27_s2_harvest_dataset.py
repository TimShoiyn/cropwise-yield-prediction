"""v27 dataset: v26 harvest-truth + Sentinel-2 red-edge/EVI/GCVI (Roadmap B1->v27).

Merges data_raw/sentinel2_indices_b1.csv (status=ok) into each v26 as-of dataset
on (field_id, year, asof_tag). Adds the red-edge contrasts that NDVI cannot capture:
  s2_ndre_minus_ndvi, s2_evi_minus_ndvi.

Rows without an S2 scene keep NaN S2 features (CatBoost handles NaN). The
companion evaluator restricts the fair ablation to rows that actually have S2.

Outputs:
  data_processed/ml_dataset_v27_s2_harvest_asof_{07_01,08_01,09_01}.csv
  reports/V27_DATASET_SUMMARY.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
ASOF_TAGS = ["07_01", "08_01", "09_01"]

S2_COLS = ["s2_ndvi_mean", "s2_ndre_mean", "s2_evi_mean", "s2_gcvi_mean",
           "s2_cloud_cover", "s2_pixels_10m", "s2_pixels_20m"]


def load_s2() -> pd.DataFrame:
    s2 = pd.read_csv(DATA_RAW / "sentinel2_indices_b1.csv")
    s2 = s2[s2["status"] == "ok"].copy()
    s2["field_id"] = s2["field_id"].astype(int)
    s2["year"] = s2["year"].astype(int)
    s2["asof_tag"] = s2["asof_tag"].astype(str)
    keep = ["field_id", "year", "asof_tag"] + [c for c in S2_COLS if c in s2.columns]
    s2 = s2[keep].drop_duplicates(["field_id", "year", "asof_tag"])
    # Red-edge / EVI contrasts vs NDVI: the extra info beyond NDVI saturation.
    s2["s2_ndre_minus_ndvi"] = s2["s2_ndre_mean"] - s2["s2_ndvi_mean"]
    s2["s2_evi_minus_ndvi"] = s2["s2_evi_mean"] - s2["s2_ndvi_mean"]
    return s2


def main() -> None:
    s2 = load_s2()
    print(f"S2 ok rows: {len(s2)}; by tag: {s2['asof_tag'].value_counts().to_dict()}")
    summary = []
    for tag in ASOF_TAGS:
        df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v26_harvest_asof_{tag}.csv")
        df["field_id"] = df["field_id"].astype(int)
        df["year"] = df["year"].astype(int)
        s2_tag = s2[s2["asof_tag"] == tag].drop(columns=["asof_tag"])
        merged = df.merge(s2_tag, on=["field_id", "year"], how="left")
        path = DATA_PROCESSED / f"ml_dataset_v27_s2_harvest_asof_{tag}.csv"
        merged.to_csv(path, index=False)
        cov = merged["s2_ndvi_mean"].notna().mean() if "s2_ndvi_mean" in merged else 0.0
        summary.append({"asof_tag": tag, "rows": len(merged),
                        "s2_coverage": f"{cov:.0%}", "cols": len(merged.columns)})
        print(f"{tag}: rows={len(merged)}, S2 coverage={cov:.0%} -> {path}")

    sm = pd.DataFrame(summary)
    md = ["# v27 dataset summary (v26 harvest + Sentinel-2)\n\n"]
    md.append("Target = real harvest (v26). Added S2 NDRE/EVI/GCVI + contrasts vs NDVI.\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V27_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
