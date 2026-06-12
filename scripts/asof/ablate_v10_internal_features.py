"""
Ablation for v10 internal features.

Compares same CatBoost params after dropping scout_* and/or soil_sample_*.
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


def drop_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = df.copy()
    scout = [c for c in out.columns if c.startswith("scout_")]
    soil = [c for c in out.columns if c.startswith("soil_sample_")]
    if variant == "v10_no_scout":
        out = out.drop(columns=scout)
    elif variant == "v10_no_soil":
        out = out.drop(columns=soil)
    elif variant == "v10_no_internal":
        out = out.drop(columns=scout + soil)
    elif variant == "v10_full":
        pass
    else:
        raise ValueError(variant)
    return out


def eval_df(df: pd.DataFrame, param_name: str = "mae_l30", use_field_id: bool = True) -> dict:
    pred = walk_forward_oof(df, PARAM_SETS[param_name], use_field_id=use_field_id)
    cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
    m = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
    return {"n": len(cmp_df), **m}


def main() -> None:
    cw = cropwise_table()
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    variants = ["v9", "v10_full", "v10_no_scout", "v10_no_soil", "v10_no_internal"]
    rows = []
    for scenario, crop_filter in scenarios:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            base9 = attach_cropwise(load_dataset("v9", tag, crop_filter), tag, cw)
            base10 = attach_cropwise(load_dataset("v10", tag, crop_filter), tag, cw)
            for variant in variants:
                if variant == "v9":
                    df = base9
                else:
                    df = drop_variant(base10, variant)
                if len(df) < 30:
                    continue
                res = eval_df(df, "mae_l30", use_field_id=True)
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "asof_label": AS_OF_LABELS[tag],
                    "variant": variant,
                    **res,
                })
            sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["asof_tag"] == tag])
            best = sub.sort_values("mape").iloc[0]
            print(f"  {tag}: best={best['variant']} MAPE={best['mape']:.1f}%")
    out = pd.DataFrame(rows)
    out_path = COMP_DIR / "asof_results_v10_ablation.csv"
    out.to_csv(out_path, index=False)

    best = out.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    md = ["# v10 Internal Feature Ablation\n\n"]
    md.append("Same model params (`mae_l30`), varying feature groups.\n\n")
    md.append("## Best by scenario/date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full ablation grid\n\n")
    md.append(out.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "asof_v10_ablation.md").write_text("".join(md), encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Saved {REPORTS / 'asof_v10_ablation.md'}")
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
