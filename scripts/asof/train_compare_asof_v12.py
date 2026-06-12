"""
v12 experiment: target from productivity_data.csv exact field-name mapping.

This fixes remaining 10x target unit errors found in low-yield years.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

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


def eval_version(version: str, tag: str, crop_filter, cw, params_name: str, use_field_id: bool = True):
    df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
    if len(df) < 25:
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
    print("TRAIN v12 — PRODUCTIVITY_DATA TARGET")
    print("=" * 80)
    cw = cropwise_table()
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    candidates = [
        ("v12_default", "default", True),
        ("v12_shallow_l10", "shallow_l10", True),
        ("v12_shallow_l30", "shallow_l30", True),
        ("v12_mae_l30", "mae_l30", True),
        ("v12_no_field_id", "shallow_l10", False),
    ]
    rows = []
    for scenario, crop_filter in scenarios:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            for model, params_name, use_field_id in candidates:
                res = eval_version("v12", tag, crop_filter, cw, params_name, use_field_id)
                if res is None:
                    continue
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "asof_label": AS_OF_LABELS[tag],
                    "model": model,
                    **res,
                })
            sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["asof_tag"] == tag])
            if not sub.empty:
                best = sub.sort_values("mape").iloc[0]
                print(f"  {tag}: best={best['model']} MAPE={best['mape']:.1f}% RMSE={best['rmse']:.3f} R2={best['r2']:.3f}; CW={best['cw_mape']:.1f}%")

    out = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v12.csv"
    out.to_csv(out_csv, index=False)
    best = out.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()

    md = ["# v12 — Productivity Data Target Override\n\n"]
    md.append("Target source: `productivity_data.csv` exact field-name match, fact yield converted from `ц/га` to `t/ha`.\n\n")
    md.append("This fixes low-yield 2020 rows where previous target was 10x too high.\n\n")
    md.append("## Best model per scenario/date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(out.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "asof_v12_results.md").write_text("".join(md), encoding="utf-8")

    print("\nFINAL BEST")
    print(best.round(3).to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {REPORTS / 'asof_v12_results.md'}")


if __name__ == "__main__":
    main()
