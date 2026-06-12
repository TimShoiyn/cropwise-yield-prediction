"""
Sprint 6 (early) — v6 minimal feature set.

Based on AUDIT.md findings:
  - drop near-duplicates (Spearman ρ ≥ 0.95)
  - drop near-zero-importance noise features (< 0.10)
  - keep one representative of each redundant cluster

We start from v4 datasets (Open-Meteo + NDVI anomaly), do NOT include v5 GDD-aligned NDVI
(it didn't add consistent value across crops/dates).

Output: data_processed/ml_dataset_clean_v6_asof_{tag}.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]

DROP_REDUNDANT = [
    "field_calculated_area",      # ρ=0.999 with field_tillable_area
    "wx_srad_mean_to_asof",       # ρ=1.000 with wx_srad_sum_to_asof
    "wx_gdd_wheat_to_asof",       # ρ=0.995 with wx_gdd_sunflower_to_asof (model has crop_id anyway)
    "ndvi_p10_max_asof",          # ρ=0.984 with ndvi_max_asof
    "wx_temp_max_last30d",        # near-perfect with monthly *_temp_max
    "wx_vpd_max_last30d",         # near-perfect with monthly *_vpd_max
    "wx_precip_sum_last30d",      # near-perfect with monthly *_precip
    "wx_srad_sum_last30d",        # near-perfect with monthly *_srad
    "wx_et0_sum_last30d",         # near-perfect with monthly *_et0
]

DROP_LOW_SIGNAL = [
    "wx_hot_d35_to_asof",         # imp ≈ 0 — too few hot-35 days here
    "wx_apr_hot_d30",             # imp ≈ 0 — none in early spring
    "wx_apr_temp_max",            # imp < 0.4 across scenarios
    "wx_apr_temp_mean",
    "wx_apr_vpd_max",
    "wx_may_hot_d30",
]


def build(tag: str):
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v4_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v6_asof_{tag}.csv"
    df = pd.read_csv(in_path)
    drop_cols = [c for c in DROP_REDUNDANT + DROP_LOW_SIGNAL if c in df.columns]
    out = df.drop(columns=drop_cols)
    out.to_csv(out_path, index=False)
    print(f"  {tag}: {df.shape[1]} → {out.shape[1]} cols (dropped {len(drop_cols)}: {drop_cols})")


def main():
    print("=" * 80)
    print("BUILD v6 MINIMAL FEATURE SET")
    print("=" * 80)
    for tag in ASOF_TAGS:
        build(tag)
    print("\nDone.")


if __name__ == "__main__":
    main()
