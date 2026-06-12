"""
Audit 2: explain R2 instability and look for model/data logic bugs.

Pooled R2 mixes three different things:
  1. between-crop level (maize ~5.4 vs lentil ~1.3 t/ha)
  2. between-year level (2020 collapse mean 0.72 vs 3.6 in 2024)
  3. within crop-year field ranking (the real local skill)

A model can get a high pooled R2 just by knowing crop means and that 2020
was bad, while having ~0 skill at ranking fields within a crop-year.

This script decomposes R2 into:
  - pooled R2
  - pooled R2 without 2020 (leverage check)
  - within-year R2
  - within-crop R2
  - within-crop-year R2 (honest field ranking skill)

Inputs:
  models_v2/asof_comparison/asof_predictions_v18_peer_transfer.csv

Outputs:
  reports/AUDIT2_R2_DECOMPOSITION.md
  reports/microscope/r2_decomposition.csv
  reports/microscope/r2_per_year.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
OUT_DIR = REPORTS / "microscope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V18 = COMP_DIR / "asof_predictions_v18_peer_transfer.csv"

SCENARIOS = ["sunflower", "wheat_combined", "all_crops"]
TAGS = ["07_01", "08_01", "09_01"]
LABEL = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
PRED_COLS = {
    "ml_peer": "pred",
    "cropwise": "cropwise_asof_t_ha",
    "naive_crop_mean": "naive_crop_mean",
}


def safe_r2(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return np.nan
    return float(r2_score(y[mask], p[mask]))


def demeaned_r2(df: pd.DataFrame, pred_col: str, group_cols: list[str] | None) -> float:
    sub = df.dropna(subset=["target_yield_t_ha", pred_col]).copy()
    if len(sub) < 4:
        return np.nan
    if group_cols is None:
        return safe_r2(sub["target_yield_t_ha"], sub[pred_col])
    y = sub["target_yield_t_ha"].astype(float)
    p = sub[pred_col].astype(float)
    grouper = [sub[c] for c in group_cols]
    y_dm = y - y.groupby(grouper).transform("mean")
    p_dm = p - p.groupby(grouper).transform("mean")
    ss_res = float(((y_dm - p_dm) ** 2).sum())
    ss_tot = float((y_dm ** 2).sum())
    if ss_tot <= 1e-9:
        return np.nan
    return 1.0 - ss_res / ss_tot


def main() -> None:
    v18 = pd.read_csv(V18)
    rows = []
    per_year_rows = []

    for scenario in SCENARIOS:
        for tag in TAGS:
            sub = v18[(v18["scenario"] == scenario) & (v18["asof_tag"] == tag)].copy()
            if len(sub) < 4:
                continue
            for model_name, col in PRED_COLS.items():
                rows.append({
                    "scenario": scenario,
                    "asof": LABEL[tag],
                    "model": model_name,
                    "n": int(sub[["target_yield_t_ha", col]].dropna().shape[0]),
                    "pooled_r2": demeaned_r2(sub, col, None),
                    "pooled_r2_no2020": demeaned_r2(sub[sub["year"] != 2020], col, None),
                    "within_year_r2": demeaned_r2(sub, col, ["year"]),
                    "within_crop_r2": demeaned_r2(sub, col, ["standard_name"]),
                    "within_crop_year_r2": demeaned_r2(sub, col, ["standard_name", "year"]),
                })
            for y, g in sub.groupby("year"):
                per_year_rows.append({
                    "scenario": scenario,
                    "asof": LABEL[tag],
                    "year": int(y),
                    "n": len(g),
                    "target_mean": float(g["target_yield_t_ha"].mean()),
                    "target_std": float(g["target_yield_t_ha"].std()),
                    "ml_r2": safe_r2(g["target_yield_t_ha"], g["pred"]),
                    "cw_r2": safe_r2(g["target_yield_t_ha"], g["cropwise_asof_t_ha"]),
                })

    decomp = pd.DataFrame(rows)
    per_year = pd.DataFrame(per_year_rows)
    decomp.to_csv(OUT_DIR / "r2_decomposition.csv", index=False)
    per_year.to_csv(OUT_DIR / "r2_per_year.csv", index=False)

    cols = ["scenario", "asof", "n", "pooled_r2", "pooled_r2_no2020", "within_year_r2", "within_crop_r2", "within_crop_year_r2"]
    show = decomp[decomp["model"] == "ml_peer"][cols]
    show_cw = decomp[decomp["model"] == "cropwise"][cols]

    md = ["# Audit 2: R2 instability decomposition\n\n"]
    md.append(
        "Pooled R2 mixes between-crop level, between-year level, and within "
        "crop-year field ranking. The honest local skill is `within_crop_year_r2`.\n\n"
    )
    md.append("## R2 decomposition (peer transfer ML)\n\n")
    md.append(show.round(3).to_markdown(index=False) + "\n\n")
    md.append("## R2 decomposition (Cropwise)\n\n")
    md.append(show_cw.round(3).to_markdown(index=False) + "\n\n")
    md.append("## Per-year R2 (2020 leverage), 1 Aug\n\n")
    md.append(per_year[per_year["asof"] == "1 Aug"].round(3).to_markdown(index=False) + "\n\n")
    md.append("## Reading guide\n\n")
    md.append(
        "- If `pooled_r2` >> `within_crop_year_r2`, R2 mostly explains crop/year level, not field ranking.\n"
        "- If `pooled_r2_no2020` differs a lot from `pooled_r2`, 2020 is a leverage point.\n"
        "- If `within_crop_year_r2` ~ 0 for everyone incl. Cropwise, nobody ranks fields inside a crop-year here.\n"
    )
    (REPORTS / "AUDIT2_R2_DECOMPOSITION.md").write_text("".join(md), encoding="utf-8")

    print("=== ml_peer decomposition ===")
    print(show.round(3).to_string(index=False))
    print("\n=== cropwise decomposition ===")
    print(show_cw.round(3).to_string(index=False))
    print(f"\nWrote {REPORTS / 'AUDIT2_R2_DECOMPOSITION.md'}")


if __name__ == "__main__":
    main()
