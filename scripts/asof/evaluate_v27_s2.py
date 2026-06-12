"""v27 evaluation: does Sentinel-2 (NDRE/EVI/GCVI) help over v26?

Fair ablation on IDENTICAL rows that have an S2 scene (+ pre-harvest filter):
  - v26_base  : v26 features WITHOUT any s2_* columns;
  - v27_s2    : v26 features + s2_* columns;
  - cropwise_asof : Cropwise estimate available by the same as-of date.

Reuses the v26 evaluator's CV/metric machinery for consistency.

Outputs:
  models_v2/asof_comparison/asof_results_v27_s2.csv
  reports/V27_S2_FINAL_RU.md
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

from scripts.asof import evaluate_v26_harvest as ev  # noqa: E402
from scripts.asof.evaluate_v26_harvest_asof_cropwise import attach_cropwise_asof  # noqa: E402

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")

S2_FEATURES = ["s2_ndvi_mean", "s2_ndre_mean", "s2_evi_mean", "s2_gcvi_mean",
               "s2_ndre_minus_ndvi", "s2_evi_minus_ndvi"]
S2_NONFEATURES = {"s2_cloud_cover", "s2_pixels_10m", "s2_pixels_20m"}


def load_dataset_v27(tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v27_s2_harvest_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= ev.MIN_TARGET_T_HA]
    if ev.PREHARVEST_ONLY and "completed_date" in df.columns:
        month, day = ev.ASOF_MD[tag]
        cd = pd.to_datetime(df["completed_date"], errors="coerce", utc=True).dt.tz_localize(None)
        asof = pd.to_datetime(dict(year=df["year"], month=month, day=day))
        df = df[cd.isna() | (cd.values > asof.values)]
    # Fair ablation: only rows that actually have an S2 scene.
    df = df[df["s2_ndvi_mean"].notna()]
    return df.reset_index(drop=True)


def predict(df: pd.DataFrame, use_s2: bool, use_field_id: bool) -> np.ndarray:
    # Build the dead-column set so S2 is included only when use_s2 is True.
    extra_dead = set(S2_NONFEATURES)
    if not use_s2:
        extra_dead |= set(S2_FEATURES)
    saved = set(ev.DEAD_OR_ID)
    try:
        ev.DEAD_OR_ID |= extra_dead
        return ev.walk_forward_predict(df, use_field_id=use_field_id)
    finally:
        ev.DEAD_OR_ID = saved


def main() -> None:
    rows, pred_rows = [], []
    for scenario, crop_filter in ev.SCENARIOS:
        print(f"\n{scenario}")
        for tag in ev.ASOF_TAGS:
            df = load_dataset_v27(tag, crop_filter)
            if len(df) < 80:
                print(f"  {tag}: only {len(df)} S2 rows -> skip")
                continue
            df["pred_v26_base"] = predict(df, use_s2=False, use_field_id=False)
            df["pred_v27_s2"] = predict(df, use_s2=True, use_field_id=False)
            df = attach_cropwise_asof(df, tag)
            fair = df[df["pred_v27_s2"].notna()].copy()

            rows.append(ev.block(df, scenario, tag, "v26_base", "pred_v26_base"))
            rows.append(ev.block(df, scenario, tag, "v27_s2", "pred_v27_s2"))
            rows.append(ev.block(fair, scenario, tag, "cropwise_asof", "cropwise_asof_t_ha"))

            tmp = df[["field_id", "year", "standard_name", "target_yield_t_ha",
                      "cropwise_asof_t_ha", "pred_v26_base", "pred_v27_s2"]].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)

            a, b, c = rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag} (n={a['n']}): base MAPE={a['mape']:.1f}% R2={a['r2']:.3f} | "
                f"s2 MAPE={b['mape']:.1f}% R2={b['r2']:.3f} wcy={b['wcy_r2']:.3f} | "
                f"CW MAPE={c['mape']:.1f}% R2={c['r2']:.3f}"
            )

    if not rows:
        print("No scenarios had enough S2 rows yet. Re-run after extraction completes.")
        return

    res = pd.DataFrame(rows)
    res.to_csv(COMP_DIR / "asof_results_v27_s2.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            COMP_DIR / "asof_predictions_v27_s2.csv", index=False
        )

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# v27 — Sentinel-2 ablation on harvest truth\n\n"]
    md.append("Fair ablation on identical S2-covered, pre-harvest rows.\n")
    md.append("`v26_base` = no S2; `v27_s2` = + NDRE/EVI/GCVI; `cropwise_asof` = same-date benchmark.\n\n")
    md.append(show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V27_S2_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote v27 results + report.")


if __name__ == "__main__":
    main()
