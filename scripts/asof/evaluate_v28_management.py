"""v28 evaluation: does management data improve v27 (harvest + S2)?

Fair ablation on identical rows:
  - v27_s2: v27 features + Sentinel-2, without v28 management columns
  - v28_mgmt: v27_s2 + as-of management features from agro_operations
  - cropwise_asof: same-date Cropwise benchmark
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


def load_dataset(tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v28_management_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= ev.MIN_TARGET_T_HA]
    if ev.PREHARVEST_ONLY and "completed_date" in df.columns:
        month, day = ev.ASOF_MD[tag]
        cd = pd.to_datetime(df["completed_date"], errors="coerce", utc=True).dt.tz_localize(None)
        asof = pd.to_datetime(dict(year=df["year"], month=month, day=day))
        df = df[cd.isna() | (cd.values > asof.values)]
    df = df[df["s2_ndvi_mean"].notna()]
    return df.reset_index(drop=True)


def predict(df: pd.DataFrame, use_mgmt: bool) -> pd.Series:
    saved = set(ev.DEAD_OR_ID)
    try:
        ev.DEAD_OR_ID |= set(S2_NONFEATURES)
        if not use_mgmt:
            ev.DEAD_OR_ID |= {c for c in df.columns if c.startswith(MGMT_PREFIX)}
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
            df["pred_v27_s2"] = predict(df, use_mgmt=False)
            df["pred_v28_mgmt"] = predict(df, use_mgmt=True)
            df = attach_cropwise_asof(df, tag)
            fair = df[df["pred_v28_mgmt"].notna()].copy()

            rows.append(ev.block(df, scenario, tag, "v27_s2", "pred_v27_s2"))
            rows.append(ev.block(df, scenario, tag, "v28_mgmt", "pred_v28_mgmt"))
            rows.append(ev.block(fair, scenario, tag, "cropwise_asof", "cropwise_asof_t_ha"))
            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_v27_s2",
                    "pred_v28_mgmt",
                ]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)

            a, b, c = rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag} (n={a['n']}): v27 MAPE={a['mape']:.1f}% R2={a['r2']:.3f} | "
                f"v28 MAPE={b['mape']:.1f}% R2={b['r2']:.3f} wcy={b['wcy_r2']:.3f} | "
                f"CW MAPE={c['mape']:.1f}% R2={c['r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    res.to_csv(COMP_DIR / "asof_results_v28_management.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(COMP_DIR / "asof_predictions_v28_management.csv", index=False)

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# v28 — management feature ablation\n\n"]
    md.append("Fair ablation on identical S2-covered, pre-harvest rows. No field_id.\n\n")
    md.append(show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V28_MANAGEMENT_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote v28 results + report.")


if __name__ == "__main__":
    main()
