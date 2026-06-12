"""v29 evaluation: pruned management features over v27 S2.

v28 showed that "all management" helps on 1 Aug for wheat/all_crops, but can
hurt barley and is mostly neutral for sunflower. This script tests a smaller,
more stable management set:

  - mgmt_application_count_asof
  - mgmt_days_since_last_application_asof
  - mgmt_n_kg_ha_asof / p2o5 / s
  - mgmt_herbicide_rate_asof / fungicide / insecticide

Fair comparison on identical S2-covered, pre-harvest rows:
  v27_s2, v28_all_mgmt, v29_pruned_mgmt, cropwise_asof.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.asof import evaluate_v26_harvest as ev  # noqa: E402
from scripts.asof.evaluate_v26_harvest_asof_cropwise import attach_cropwise_asof  # noqa: E402
from scripts.asof.evaluate_v27_s2 import S2_NONFEATURES  # noqa: E402

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")

MGMT_PREFIX = "mgmt_"
PRUNED_MGMT_FEATURES = {
    "mgmt_application_count_asof",
    "mgmt_days_since_last_application_asof",
    "mgmt_n_kg_ha_asof",
    "mgmt_p2o5_kg_ha_asof",
    "mgmt_s_kg_ha_asof",
    "mgmt_herbicide_rate_asof",
    "mgmt_fungicide_rate_asof",
    "mgmt_insecticide_rate_asof",
}


def load_dataset(tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v28_management_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= ev.MIN_TARGET_T_HA]
    if ev.PREHARVEST_ONLY and "completed_date" in df.columns:
        month, day = ev.ASOF_MD[tag]
        completed = pd.to_datetime(df["completed_date"], errors="coerce", utc=True).dt.tz_localize(None)
        asof = pd.to_datetime(dict(year=df["year"], month=month, day=day))
        df = df[completed.isna() | (completed.values > asof.values)]
    df = df[df["s2_ndvi_mean"].notna()]
    return df.reset_index(drop=True)


def predict(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode not in {"v27_s2", "v28_all_mgmt", "v29_pruned_mgmt"}:
        raise ValueError(f"Unknown mode: {mode}")

    all_mgmt = {c for c in df.columns if c.startswith(MGMT_PREFIX)}
    saved = set(ev.DEAD_OR_ID)
    try:
        ev.DEAD_OR_ID |= set(S2_NONFEATURES)
        if mode == "v27_s2":
            ev.DEAD_OR_ID |= all_mgmt
        elif mode == "v29_pruned_mgmt":
            ev.DEAD_OR_ID |= all_mgmt - PRUNED_MGMT_FEATURES
        return ev.walk_forward_predict(df, use_field_id=False)
    finally:
        ev.DEAD_OR_ID = saved


def main() -> None:
    rows, pred_rows = [], []
    for scenario, crop_filter in ev.SCENARIOS:
        print(f"\n{scenario}")
        for tag in ev.ASOF_TAGS:
            df = load_dataset(tag, crop_filter)
            if len(df) < 80:
                print(f"  {tag}: only {len(df)} rows -> skip")
                continue
            df["pred_v27_s2"] = predict(df, "v27_s2")
            df["pred_v28_all_mgmt"] = predict(df, "v28_all_mgmt")
            df["pred_v29_pruned_mgmt"] = predict(df, "v29_pruned_mgmt")
            df = attach_cropwise_asof(df, tag)

            fair = df[df["pred_v29_pruned_mgmt"].notna()].copy()
            rows.append(ev.block(df, scenario, tag, "v27_s2", "pred_v27_s2"))
            rows.append(ev.block(df, scenario, tag, "v28_all_mgmt", "pred_v28_all_mgmt"))
            rows.append(ev.block(df, scenario, tag, "v29_pruned_mgmt", "pred_v29_pruned_mgmt"))
            rows.append(ev.block(fair, scenario, tag, "cropwise_asof", "cropwise_asof_t_ha"))

            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_v27_s2",
                    "pred_v28_all_mgmt",
                    "pred_v29_pruned_mgmt",
                ]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)

            a, b, c, d = rows[-4], rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag} (n={a['n']}): "
                f"v27 MAPE={a['mape']:.1f}% R2={a['r2']:.3f} | "
                f"v28 MAPE={b['mape']:.1f}% R2={b['r2']:.3f} | "
                f"v29 MAPE={c['mape']:.1f}% R2={c['r2']:.3f} wcy={c['wcy_r2']:.3f} | "
                f"CW MAPE={d['mape']:.1f}% R2={d['r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    res.to_csv(COMP_DIR / "asof_results_v29_pruned_management.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            COMP_DIR / "asof_predictions_v29_pruned_management.csv", index=False
        )

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# v29 — pruned management feature ablation\n\n"]
    md.append("Fair ablation on identical S2-covered, pre-harvest rows. No field_id.\n\n")
    md.append("Pruned features:\n\n")
    for feat in sorted(PRUNED_MGMT_FEATURES):
        md.append(f"- `{feat}`\n")
    md.append("\n")
    md.append(
        show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(
            index=False
        )
    )
    md.append("\n")
    (REPORTS / "V29_PRUNED_MANAGEMENT_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote v29 results + report.")


if __name__ == "__main__":
    main()
