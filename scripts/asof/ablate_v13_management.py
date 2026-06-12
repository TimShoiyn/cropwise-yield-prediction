"""
v13 management ablation.

Tests compact subsets of management features to avoid sparse-feature overfit.
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

SCENARIOS = [
    ("sunflower", ["sunflower"]),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("all_crops", None),
]

COUNT_KEEP = [
    "mgmt_ops_n_asof",
    "mgmt_has_ops_asof",
    "mgmt_op_application_n",
    "mgmt_op_soil_n",
    "mgmt_op_other_n",
    "mgmt_subtype_harrowing_n",
    "mgmt_subtype_discing_n",
]
RATE_KEEP = [
    "mgmt_fertilizer_items_n",
    "mgmt_fertilizer_fact_rate_sum",
    "mgmt_fertilizer_fact_rate_max",
    "mgmt_seed_items_n",
    "mgmt_seed_fact_rate_sum",
    "mgmt_seed_fact_rate_max",
    "mgmt_chemical_items_n",
    "mgmt_chemical_fact_rate_sum",
]
NUTRIENT_KEEP = [
    "mgmt_fert_N_rate_sum",
    "mgmt_fert_P2O5_rate_sum",
    "mgmt_fert_K2O_rate_sum",
    "mgmt_fert_S_rate_sum",
    "mgmt_fert_Mg_rate_sum",
]
FLAG_KEEP = [c for c in []]


def variant_df(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    mgmt_cols = [c for c in df.columns if c.startswith("mgmt_")]
    if variant == "v12_no_mgmt":
        return df.drop(columns=mgmt_cols, errors="ignore")
    if variant == "v13_full":
        return df
    keep = set()
    if variant in {"counts", "counts_rates", "counts_rates_nutrients"}:
        keep |= set(COUNT_KEEP)
    if variant in {"rates", "counts_rates", "counts_rates_nutrients"}:
        keep |= set(RATE_KEEP)
    if variant in {"nutrients", "counts_rates_nutrients"}:
        keep |= set(NUTRIENT_KEEP)
    drop = [c for c in mgmt_cols if c not in keep]
    return df.drop(columns=drop, errors="ignore")


def eval_df(df: pd.DataFrame, params_name: str = "shallow_l30", use_field_id: bool = True):
    pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
    cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
    if cmp_df.empty:
        return None
    m = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
    return {"n": len(cmp_df), **m}


def main() -> None:
    cw = cropwise_table()
    variants = ["v12_no_mgmt", "v13_full", "counts", "rates", "nutrients", "counts_rates", "counts_rates_nutrients"]
    rows = []
    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            df = attach_cropwise(load_dataset("v13", tag, crop_filter), tag, cw)
            if len(df) < 25:
                continue
            for variant in variants:
                for params_name, use_field_id, suffix in [("shallow_l30", True, "l30"), ("shallow_l10", False, "nofield")]:
                    vd = variant_df(df, variant)
                    res = eval_df(vd, params_name=params_name, use_field_id=use_field_id)
                    if res is None:
                        continue
                    rows.append({
                        "scenario": scenario,
                        "asof_tag": tag,
                        "asof_label": AS_OF_LABELS[tag],
                        "variant": variant,
                        "config": suffix,
                        **res,
                    })
            sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["asof_tag"] == tag])
            best = sub.sort_values("mape").iloc[0]
            print(f"  {tag}: best={best['variant']}:{best['config']} MAPE={best['mape']:.1f}%")
    out = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v13_ablation.csv"
    out.to_csv(out_csv, index=False)
    best = out.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    md = ["# v13 Management Feature Ablation\n\n"]
    md.append("Compact management subsets vs full v13.\n\n")
    md.append("## Best by scenario/date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(out.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "asof_v13_ablation.md").write_text("".join(md), encoding="utf-8")
    print(f"\nSaved {out_csv}")
    print(f"Saved {REPORTS / 'asof_v13_ablation.md'}")
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
