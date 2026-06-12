"""
v16 conflict-clean sensitivity.

Filters out field/year rows where productivity_data target and yield_maps mean
differ by >1 t/ha. Builds filtered variants for v12, v14, v15_rates.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
ASOF_TAGS = ["07_01", "08_01", "09_01"]
VERSIONS = ["v12", "v14", "v15_rates"]


def main() -> None:
    conflicts = pd.read_csv(REPORTS / "target_conflicts_productivity_vs_yield_maps.csv")
    conflicts = conflicts[conflicts["conflict_gt_1t"]].copy()
    bad = set(zip(conflicts["field_id"].astype(int), conflicts["year"].astype(int)))
    print("=" * 80)
    print("BUILD v16 — CONFLICT-CLEAN SENSITIVITY")
    print("=" * 80)
    print(f"Conflict rows removed: {len(bad)}")
    print(conflicts[["field_id", "field_name", "year", "prod_fact_t_ha", "yield_map_mean_t_ha", "abs_diff"]].round(3).to_string(index=False))
    for version in VERSIONS:
        out_version = f"v16_{version}"
        for tag in ASOF_TAGS:
            path = DATA_PROCESSED / f"ml_dataset_clean_{version}_asof_{tag}.csv"
            df = pd.read_csv(path)
            mask = ~df.apply(lambda r: (int(r["field_id"]), int(r["year"])) in bad, axis=1)
            out = df[mask].copy()
            out_path = DATA_PROCESSED / f"ml_dataset_clean_{out_version}_asof_{tag}.csv"
            out.to_csv(out_path, index=False)
            print(f"  {version}->{out_version} {tag}: {len(df)} -> {len(out)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
