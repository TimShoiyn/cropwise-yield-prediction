"""
A1 audit: low-yield target rows in v24.

Rows with target < 1 t/ha distort MAPE and often represent a different
problem (crop failure, abandoned field, partial harvest, unit/target issue)
rather than ordinary yield prediction. This script classifies them using:
  - target source and crop/year distribution;
  - NDVI growth profile;
  - as-of Cropwise forecast error;
  - sowing/harvest metadata.

Outputs:
  reports/A1_LOW_YIELD_AUDIT_RU.md
  reports/microscope/a1_low_yield_rows.csv
  reports/microscope/a1_low_yield_summary_by_crop_year.csv
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

from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
MICRO = REPORTS / "microscope"
MICRO.mkdir(parents=True, exist_ok=True)

ASOF_TAG = "08_01"
LOW_TARGET = 1.0


def classify_row(row: pd.Series) -> str:
    ndvi_max = row.get("ndvi_max_asof")
    ndvi_n = row.get("ndvi_n_obs_asof")
    target_source = str(row.get("target_source", ""))
    cropwise = row.get("cropwise_asof_t_ha")
    target = row.get("target_yield_t_ha")

    if pd.isna(ndvi_n) or ndvi_n < 3:
        return "no_satellite_evidence"
    if pd.notna(ndvi_max) and ndvi_max < 0.35:
        return "likely_crop_failure_low_ndvi"
    if pd.notna(ndvi_max) and ndvi_max >= 0.55 and pd.notna(cropwise) and cropwise >= 1.5:
        return "suspect_target_high_ndvi_and_cropwise"
    if pd.notna(ndvi_max) and ndvi_max >= 0.55:
        return "suspect_target_high_ndvi"
    if "productivity_t_ha" in target_source and pd.notna(target) and target < 0.3:
        return "suspect_productivity_fallback_tiny"
    return "ambiguous_low_yield"


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    cw = cropwise_table()
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v24_agro_asof_{ASOF_TAG}.csv")
    df = attach_cropwise(df, ASOF_TAG, cw)
    df = df.dropna(subset=["target_yield_t_ha"]).copy()

    low = df[df["target_yield_t_ha"] < LOW_TARGET].copy()
    normal = df[df["target_yield_t_ha"] >= LOW_TARGET].copy()

    low["low_yield_class"] = low.apply(classify_row, axis=1)
    low["cropwise_abs_error"] = (low["cropwise_asof_t_ha"] - low["target_yield_t_ha"]).abs()
    low["cropwise_over_target_ratio"] = low["cropwise_asof_t_ha"] / low["target_yield_t_ha"].replace(0, np.nan)

    out_cols = [
        "field_id",
        "year",
        "standard_name",
        "target_yield_t_ha",
        "target_source",
        "low_yield_class",
        "cropwise_asof_t_ha",
        "cropwise_abs_error",
        "ndvi_n_obs_asof",
        "ndvi_max_asof",
        "ndvi_mean_asof",
        "ndvi_integral_asof",
        "cw_sm_mean_asof",
        "gdd_to_asof",
        "heat_days_ge30_to_asof",
        "sowing_date",
        "harvesting_date",
        "field_name",
    ]
    low[[c for c in out_cols if c in low.columns]].to_csv(MICRO / "a1_low_yield_rows.csv", index=False)

    summary = (
        low.groupby(["standard_name", "year"], as_index=False)
        .agg(
            low_rows=("target_yield_t_ha", "size"),
            target_mean=("target_yield_t_ha", "mean"),
            ndvi_max_mean=("ndvi_max_asof", "mean"),
            cropwise_mean=("cropwise_asof_t_ha", "mean"),
        )
        .sort_values(["low_rows", "standard_name", "year"], ascending=[False, True, True])
    )
    summary.to_csv(MICRO / "a1_low_yield_summary_by_crop_year.csv", index=False)

    class_counts = low["low_yield_class"].value_counts().reset_index()
    source_counts = low["target_source"].value_counts().reset_index()
    crop_counts = low["standard_name"].value_counts().head(12).reset_index()
    year_counts = low["year"].value_counts().sort_index().reset_index()

    normal_ndvi = normal["ndvi_max_asof"].describe(percentiles=[0.25, 0.5, 0.75])
    low_ndvi = low["ndvi_max_asof"].describe(percentiles=[0.25, 0.5, 0.75])

    suspect = low[low["low_yield_class"].str.startswith("suspect")].copy()
    likely_failure = low[low["low_yield_class"].eq("likely_crop_failure_low_ndvi")].copy()

    md = ["# A1 low-yield target audit (v24)\n\n"]
    md.append(f"Dataset: `ml_dataset_v24_agro_asof_{ASOF_TAG}.csv` (1 Aug snapshot).\n\n")
    md.append("## Main counts\n\n")
    md.append(f"- Total target rows: **{len(df):,}**\n")
    md.append(f"- Normal rows (`target >= {LOW_TARGET}`): **{len(normal):,}** ({fmt_pct(len(normal) / len(df))})\n")
    md.append(f"- Low-yield rows (`target < {LOW_TARGET}`): **{len(low):,}** ({fmt_pct(len(low) / len(df))})\n\n")

    md.append("## Classification\n\n")
    md.append(class_counts.to_markdown(index=False))
    md.append("\n\n")

    md.append("Interpretation:\n\n")
    md.append("- `likely_crop_failure_low_ndvi`: low target is agronomically plausible; NDVI never formed a strong canopy.\n")
    md.append("- `suspect_target_high_ndvi*`: target is suspicious; field looked productive by NDVI and/or Cropwise expected a normal crop.\n")
    md.append("- `ambiguous_low_yield`: not enough evidence to trust or reject; keep separate from main yield metric.\n\n")

    md.append("## By crop\n\n")
    md.append(crop_counts.to_markdown(index=False))
    md.append("\n\n## By year\n\n")
    md.append(year_counts.to_markdown(index=False))
    md.append("\n\n## Target source among low-yield rows\n\n")
    md.append(source_counts.to_markdown(index=False))
    md.append("\n\n")

    md.append("## NDVI sanity\n\n")
    md.append("`ndvi_max_asof` distribution:\n\n")
    ndvi_table = pd.DataFrame(
        {
            "metric": ["count", "mean", "p25", "median", "p75", "max"],
            "low_yield": [
                low_ndvi.get("count"),
                low_ndvi.get("mean"),
                low_ndvi.get("25%"),
                low_ndvi.get("50%"),
                low_ndvi.get("75%"),
                low_ndvi.get("max"),
            ],
            "normal": [
                normal_ndvi.get("count"),
                normal_ndvi.get("mean"),
                normal_ndvi.get("25%"),
                normal_ndvi.get("50%"),
                normal_ndvi.get("75%"),
                normal_ndvi.get("max"),
            ],
        }
    )
    md.append(ndvi_table.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Decision policy for training/evaluation\n\n")
    md.append("1. Main yield-regression benchmark should use `target >= 1 t/ha`.\n")
    md.append("2. Low-yield rows should not be mixed into MAPE because they represent a different task and can explode percentage error.\n")
    md.append("3. Keep low-yield rows as a separate **crop failure / anomaly detection** problem.\n")
    md.append("4. For dissertation claims, report both:\n")
    md.append("   - main regression on normal-yield rows;\n")
    md.append("   - separate low-yield audit coverage and examples.\n\n")

    md.append("## Key numbers\n\n")
    md.append(f"- Likely real crop failures: **{len(likely_failure)}**\n")
    md.append(f"- Suspect target rows with high NDVI evidence: **{len(suspect)}**\n")
    md.append(f"- Detailed rows: `reports/microscope/a1_low_yield_rows.csv`\n")
    md.append(f"- Crop-year summary: `reports/microscope/a1_low_yield_summary_by_crop_year.csv`\n")

    (REPORTS / "A1_LOW_YIELD_AUDIT_RU.md").write_text("".join(md), encoding="utf-8")
    print(f"low rows={len(low)} normal={len(normal)} suspect={len(suspect)} likely_failure={len(likely_failure)}")
    print("wrote reports/A1_LOW_YIELD_AUDIT_RU.md")


if __name__ == "__main__":
    main()
