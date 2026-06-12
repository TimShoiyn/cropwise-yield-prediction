"""
v15_rates = v12 + rates-only compact management subset.

This is designed for wheat, based on v13 ablation where the `rates` subset
improved wheat 1 Aug. It keeps only low-dimensional operation/mix rates and
nutrient proxy sums.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]

KEEP_MGMT_RATES = [
    "mgmt_fertilizer_items_n",
    "mgmt_fertilizer_fact_rate_sum",
    "mgmt_seed_items_n",
    "mgmt_seed_fact_rate_sum",
    "mgmt_chemical_items_n",
    "mgmt_chemical_fact_rate_sum",
    "mgmt_fert_N_rate_sum",
    "mgmt_fert_P2O5_rate_sum",
    "mgmt_fert_K2O_rate_sum",
    "mgmt_fert_S_rate_sum",
    "mgmt_fert_Mg_rate_sum",
]


def main() -> None:
    print("=" * 80)
    print("BUILD v15_rates — RATES-ONLY MANAGEMENT")
    print("=" * 80)
    for tag in ASOF_TAGS:
        v12 = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v12_asof_{tag}.csv")
        v13 = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v13_asof_{tag}.csv")
        keep = ["field_id", "year"] + [c for c in KEEP_MGMT_RATES if c in v13.columns]
        mgmt = v13[keep].copy()
        out = v12.merge(mgmt, on=["field_id", "year"], how="left")
        out_path = DATA_PROCESSED / f"ml_dataset_clean_v15_rates_asof_{tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"  {tag}: {len(out)} rows, {v12.shape[1]} -> {out.shape[1]} cols (+{out.shape[1]-v12.shape[1]})")
    print("\nDone.")


if __name__ == "__main__":
    main()
