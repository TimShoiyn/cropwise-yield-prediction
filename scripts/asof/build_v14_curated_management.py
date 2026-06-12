"""
v14 = v12 + curated compact management features selected from v13 ablation.

Keeps only stable low-dimensional management groups:
  - operation counts
  - fertilizer/seed/chemical count/rate totals
  - nutrient proxy sums N/P2O5/K2O/S/Mg

Drops sparse dates/max/amount/custom flag features from full v13.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]

KEEP_MGMT = [
    "mgmt_ops_n_asof",
    "mgmt_has_ops_asof",
    "mgmt_op_application_n",
    "mgmt_op_soil_n",
    "mgmt_op_other_n",
    "mgmt_subtype_harrowing_n",
    "mgmt_subtype_discing_n",
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
    print("BUILD v14 — CURATED MANAGEMENT FEATURES")
    print("=" * 80)
    for tag in ASOF_TAGS:
        v12 = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v12_asof_{tag}.csv")
        v13 = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v13_asof_{tag}.csv")
        keep = ["field_id", "year"] + [c for c in KEEP_MGMT if c in v13.columns]
        mgmt = v13[keep].copy()
        out = v12.merge(mgmt, on=["field_id", "year"], how="left")
        out_path = DATA_PROCESSED / f"ml_dataset_clean_v14_asof_{tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"  {tag}: {len(out)} rows, {v12.shape[1]} -> {out.shape[1]} cols (+{out.shape[1]-v12.shape[1]})")
    print("\nDone.")


if __name__ == "__main__":
    main()
