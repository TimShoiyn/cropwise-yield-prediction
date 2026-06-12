"""B2.1: evaluate v26 against a fair Cropwise as-of benchmark.

`evaluate_v26_harvest.py` compared against `cropwise_t_ha`, which is the latest
available/final Cropwise estimate. This script compares against
`cropwise_asof_t_ha` built from productivity_estimate_histories:
latest Cropwise estimate <= 1 Jul / 1 Aug / 1 Sep.

This is the fairer benchmark for the thesis because both systems see only
information available by the same date.
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

DATA_CLEAN = Path("data_clean")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")


def attach_cropwise_asof(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    cw = pd.read_csv(DATA_CLEAN / "cropwise_estimates_asof_clean.csv")
    cw = cw[cw["asof_tag"] == tag][
        ["field_id", "year", "cropwise_asof_t_ha", "cropwise_history_date"]
    ].copy()
    cw["field_id"] = cw["field_id"].astype(int)
    cw["year"] = cw["year"].astype(int)
    out = df.merge(cw, on=["field_id", "year"], how="left")
    return out


def main() -> None:
    rows, pred_rows = [], []
    for scenario, crop_filter in ev.SCENARIOS:
        print(f"\n{scenario}")
        for tag in ev.ASOF_TAGS:
            # Predict before attaching as-of Cropwise, so the benchmark cannot leak as a feature.
            df = ev.load_dataset(tag, crop_filter)
            df["pred_v26_full"] = ev.walk_forward_predict(df, use_field_id=True)
            df["pred_v26_no_field"] = ev.walk_forward_predict(df, use_field_id=False)
            df = attach_cropwise_asof(df, tag)

            fair = df[df["pred_v26_full"].notna()].copy()
            rows.append(ev.block(df, scenario, tag, "v26_full", "pred_v26_full"))
            rows.append(ev.block(df, scenario, tag, "v26_no_field", "pred_v26_no_field"))
            rows.append(ev.block(fair, scenario, tag, "cropwise_asof", "cropwise_asof_t_ha"))
            rows.append(ev.block(fair, scenario, tag, "cropwise_final", "cropwise_t_ha"))

            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "cropwise_history_date",
                    "cropwise_t_ha",
                    "pred_v26_full",
                    "pred_v26_no_field",
                ]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)

            a, b, c, d = rows[-4], rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag}: v26_nf MAPE={b['mape']:.1f}% R2={b['r2']:.3f} | "
                f"CW_asof MAPE={c['mape']:.1f}% R2={c['r2']:.3f} | "
                f"CW_final MAPE={d['mape']:.1f}% R2={d['r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v26_harvest_asof_cropwise.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v26_harvest_asof_cropwise.csv", index=False)

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))

    md = ["# B2.1 — v26 vs Cropwise as-of benchmark\n\n"]
    md.append("Target = real combine harvest, target >= 1 t/ha. Validation = walk-forward by year.\n\n")
    md.append("`cropwise_asof` = latest Cropwise history value on/before the same as-of date. ")
    md.append("`cropwise_final` is shown only as an upper/reference line and is not a fair as-of benchmark.\n\n")
    md.append(
        show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(
            index=False
        )
    )
    md.append("\n")
    (REPORTS / "B21_V26_ASOF_CROPWISE_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote B2.1 results + report.")


if __name__ == "__main__":
    main()
