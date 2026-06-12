"""
v17 strict-target policy comparison.

Trains the same candidate policy as v15/v16 on the v17 datasets:
    - v17 full  : every row, target re-prioritized to physical_t_ha > prod_fact_t_ha
    - v17 clean : conflict rows (physical vs prod_fact disagree by >1 t/ha) dropped

Reports MAPE, MAE, RMSE, R^2 for ML and Cropwise per scenario / as-of date.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import scripts.asof.train_compare_asof_v9 as base  # noqa: E402

base.BASE_NON_FEATURE = base.BASE_NON_FEATURE | {
    "target_v17",
    "target_v17_source",
    "target_v17_conflict",
    "physical_t_ha",
    "prod_fact_t_ha",
    "productivity_t_ha",
}

from scripts.asof.train_compare_asof_v9 import (  # noqa: E402
    AS_OF_TAGS,
    AS_OF_LABELS,
    COMP_DIR,
    REPORTS,
    PARAM_SETS,
    attach_cropwise,
    cropwise_table,
    load_dataset,
    metrics,
    walk_forward_oof,
)

SCENARIOS = [
    ("sunflower", ["sunflower"]),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("all_crops", None),
]

CANDIDATES = [
    ("v17_v12_full", "v17_v12_full_default", "default", True),
    ("v17_v12_full", "v17_v12_full_l30", "shallow_l30", True),
    ("v17_v12_full", "v17_v12_full_no_fid", "shallow_l10", False),
    ("v17_v12_clean", "v17_v12_clean_default", "default", True),
    ("v17_v12_clean", "v17_v12_clean_l30", "shallow_l30", True),
    ("v17_v12_clean", "v17_v12_clean_no_fid", "shallow_l10", False),
    ("v17_v14_full", "v17_v14_full_l10", "shallow_l10", True),
    ("v17_v14_full", "v17_v14_full_l30", "shallow_l30", True),
    ("v17_v14_clean", "v17_v14_clean_l10", "shallow_l10", True),
    ("v17_v14_clean", "v17_v14_clean_l30", "shallow_l30", True),
    ("v17_v15_rates_full", "v17_rates_full_l10", "shallow_l10", True),
    ("v17_v15_rates_full", "v17_rates_full_l30", "shallow_l30", True),
    ("v17_v15_rates_clean", "v17_rates_clean_l10", "shallow_l10", True),
    ("v17_v15_rates_clean", "v17_rates_clean_l30", "shallow_l30", True),
]


def eval_candidate(version, tag, crop_filter, cw, params_name, use_field_id):
    df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
    if len(df) < 20:
        return None
    pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
    cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
    if cmp_df.empty:
        return None
    cmp_cw = cmp_df.dropna(subset=["cropwise_asof_t_ha"])
    ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
    cw_m = metrics(cmp_cw["target_yield_t_ha"], cmp_cw["cropwise_asof_t_ha"]) if len(cmp_cw) else {}
    return {
        "n_ml": len(cmp_df),
        "n_cw": len(cmp_cw),
        "mape": ml["mape"],
        "rmse": ml["rmse"],
        "mae": ml["mae"],
        "r2": ml["r2"],
        "cw_mape": cw_m.get("mape", float("nan")),
        "cw_rmse": cw_m.get("rmse", float("nan")),
        "cw_mae": cw_m.get("mae", float("nan")),
        "cw_r2": cw_m.get("r2", float("nan")),
    }


def main() -> None:
    print("=" * 80)
    print("TRAIN v17 — STRICT TARGET POLICY")
    print("=" * 80)
    cw = cropwise_table()
    rows = []
    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            for version, model_label, params_name, use_field_id in CANDIDATES:
                res = eval_candidate(version, tag, crop_filter, cw, params_name, use_field_id)
                if res is None:
                    continue
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "asof_label": AS_OF_LABELS[tag],
                    "version": version,
                    "model": model_label,
                    **res,
                })
            sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["asof_tag"] == tag])
            if sub.empty:
                print(f"  {tag}: no result")
                continue
            best = sub.sort_values("mape").iloc[0]
            print(
                f"  {tag}: best={best['model']} MAPE={best['mape']:.1f}% RMSE={best['rmse']:.2f} "
                f"MAE={best['mae']:.2f} R2={best['r2']:.2f} | CW MAPE={best['cw_mape']:.1f}% "
                f"R2={best['cw_r2']:.2f} n={int(best['n_ml'])}"
            )

    summary = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v17_strict.csv"
    summary.to_csv(out_csv, index=False)

    best = (
        summary.sort_values(["scenario", "asof_tag", "mape"])
        .groupby(["scenario", "asof_tag"], as_index=False)
        .first()
    )
    md = ["# v17 Strict Target Policy Comparison\n\n"]
    md.append(
        "Target rebuild: physical_t_ha > prod_fact_t_ha > productivity_t_ha. "
        "Conflict (physical vs prod_fact disagree by > 1 t/ha) flagged. "
        "`full` = every row kept, `clean` = conflicts dropped.\n\n"
    )
    md.append("## Best policy per scenario / as-of date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "asof_v17_strict_results.md").write_text("".join(md), encoding="utf-8")

    print("\nFINAL BEST")
    print(best.round(3).to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {REPORTS / 'asof_v17_strict_results.md'}")


if __name__ == "__main__":
    main()
