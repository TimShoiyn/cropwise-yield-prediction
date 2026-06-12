"""
v11 sensitivity: remove rows where combine yield maps strongly disagree with factual target.

Outputs:
  - ml_dataset_clean_v11_asof_{tag}.csv   = v9 minus suspect target rows
  - ml_dataset_clean_v11i_asof_{tag}.csv  = v10 minus suspect target rows
  - reports/v11_removed_yieldmap_target_suspects.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
ASOF_TAGS = ["07_01", "08_01", "09_01"]
DIFF_THRESHOLD_T_HA = 1.0


def pick_yield_value(row: pd.Series) -> float:
    for col in ["external_average", "totals.result.average.value"]:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value) and 0.1 <= float(value) <= 12.0:
            return float(value)
    value = pd.to_numeric(row.get("calculated_average"), errors="coerce")
    units = str(row.get("units", ""))
    if pd.notna(value):
        value = float(value)
        if units == "tonn_per_ha" and 0.1 <= value <= 12.0:
            return value
        if units == "tonn_per_acre" and 0.1 <= value <= 6.0:
            return value * 2.47105381
    return np.nan


def suspect_pairs() -> pd.DataFrame:
    ym = pd.read_csv(DATA_RAW / "yield_maps.csv", low_memory=False)
    ym["created_at"] = pd.to_datetime(ym["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    ym["year"] = ym["created_at"].dt.year
    ym["yield_map_t_ha"] = ym.apply(pick_yield_value, axis=1)
    ym = ym.dropna(subset=["field_id", "year", "yield_map_t_ha"]).copy()
    ym["field_id"] = ym["field_id"].astype(int)
    ym["year"] = ym["year"].astype(int)
    ym = ym[(ym["yield_map_t_ha"] >= 0.2) & (ym["yield_map_t_ha"] <= 10.0)]
    agg = ym.groupby(["field_id", "year"]).agg(
        yield_map_mean_t_ha=("yield_map_t_ha", "mean"),
        yield_map_n=("yield_map_t_ha", "size"),
    ).reset_index()

    target = pd.read_csv(DATA_PROCESSED / "targets_factual_t_ha.csv")
    target = target[target["kept"]].copy()
    target = target[["field_id", "year", "standard_name", "target_yield_t_ha", "target_source"]]
    cmp = target.merge(agg, on=["field_id", "year"], how="inner")
    cmp["diff_map_minus_target"] = cmp["yield_map_mean_t_ha"] - cmp["target_yield_t_ha"]
    cmp["abs_diff"] = cmp["diff_map_minus_target"].abs()
    return cmp[cmp["abs_diff"] > DIFF_THRESHOLD_T_HA].sort_values("abs_diff", ascending=False)


def filter_dataset(version_in: str, version_out: str, tag: str, suspects: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_{version_in}_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_{version_out}_asof_{tag}.csv"
    df = pd.read_csv(in_path)
    bad = set(zip(suspects["field_id"].astype(int), suspects["year"].astype(int)))
    keep_mask = ~df.apply(lambda r: (int(r["field_id"]), int(r["year"])) in bad, axis=1)
    out = df[keep_mask].copy()
    out.to_csv(out_path, index=False)
    print(f"  {version_in}->{version_out} {tag}: {len(df)} -> {len(out)} rows (removed {len(df)-len(out)})")


def main() -> None:
    print("=" * 80)
    print("BUILD v11 — YIELD MAP TARGET CLEAN SENSITIVITY")
    print("=" * 80)
    suspects = suspect_pairs()
    out_suspects = REPORTS / "v11_removed_yieldmap_target_suspects.csv"
    suspects.to_csv(out_suspects, index=False)
    print(f"Suspect target rows: {len(suspects)} (> {DIFF_THRESHOLD_T_HA} t/ha)")
    print(suspects[["field_id", "year", "standard_name", "target_yield_t_ha", "yield_map_mean_t_ha", "abs_diff"]].round(3).to_string(index=False))
    print(f"Saved {out_suspects}")
    for tag in ASOF_TAGS:
        filter_dataset("v9", "v11", tag, suspects)
        filter_dataset("v10", "v11i", tag, suspects)
    print("\nDone.")


if __name__ == "__main__":
    main()
