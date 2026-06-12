"""
Sprint 6.1 — v8 = v7 (factual targets) + SoilGrids 250m soil properties.

SoilGrids is a global ML soil map at 250m resolution. We use 3 depth layers
in the rooting zone:
  - 0-5cm   (topsoil)
  - 5-15cm  (plough layer)
  - 15-30cm (subsoil)

Properties: clay, sand, silt, soc, nitrogen, bdod, cec, phh2o.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
ASOF_TAGS = ["07_01", "08_01", "09_01"]


def main():
    print("=" * 80)
    print("BUILD v8 = v7 + SoilGrids")
    print("=" * 80)
    sg = pd.read_csv(DATA_RAW / "soilgrids.csv")
    sg_cols = [c for c in sg.columns if c.startswith("sg_")]
    print(f"  SoilGrids: {len(sg)} fields, {len(sg_cols)} features")
    sg_keep = ["field_id"] + sg_cols
    sg = sg[sg_keep]

    for tag in ASOF_TAGS:
        in_path = DATA_PROCESSED / f"ml_dataset_clean_v7_asof_{tag}.csv"
        out_path = DATA_PROCESSED / f"ml_dataset_clean_v8_asof_{tag}.csv"
        df = pd.read_csv(in_path)
        n0 = df.shape[1]
        out = df.merge(sg, on="field_id", how="left")
        cov = out[sg_cols].notna().mean().mean()
        out.to_csv(out_path, index=False)
        print(f"  {tag}: {n0} → {out.shape[1]} cols (+{out.shape[1]-n0}), {len(out)} rows, sg coverage avg {cov*100:.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
