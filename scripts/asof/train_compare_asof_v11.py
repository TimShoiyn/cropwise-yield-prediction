"""
v11 sensitivity: yield-map target-clean datasets.

Compares v9/v10 vs v11/v11i after removing 8 rows where combine yield map
and factual target disagree by >1 t/ha.
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
    print("TRAIN v11 — YIELD MAP TARGET-CLEAN SENSITIVITY")
    print("=" * 80)
    cw = cropwise_table()
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    candidates = [
        ("v9", "v9_mae_l30", "mae_l30", True),
        ("v10", "v10_mae_l30", "mae_l30", True),
        ("v11", "v11_clean_mae_l30", "mae_l30", True),
        ("v11i", "v11i_clean_internal_mae_l30", "mae_l30", True),
        ("v11", "v11_clean_no_field_id", "shallow_l10", False),
        ("v11i", "v11i_clean_internal_no_field_id", "shallow_l10", False),
    ]
    rows = []
    for scenario, crop_filter in scenarios:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            for version, model, params_name, use_field_id in candidates:
                res = eval_version(version, tag, crop_filter, cw, params_name, use_field_id)
                if res is None:
                    continue
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "asof_label": AS_OF_LABELS[tag],
                    "version": version,
                    "model": model,
                    **res,
                })
            sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["asof_tag"] == tag])
            if not sub.empty:
                best = sub.sort_values("mape").iloc[0]
                clean = sub[sub["version"].isin(["v11", "v11i"])].sort_values("mape").head(1)
                clean_txt = f"cleanbest={clean.iloc[0]['model']} {clean.iloc[0]['mape']:.1f}%" if not clean.empty else "cleanbest=nan"
                print(f"  {tag}: best={best['model']} {best['mape']:.1f}%; {clean_txt}; CW={best['cw_mape']:.1f}%")

    out = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v11.csv"
    out.to_csv(out_csv, index=False)
    best = out.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    clean_best = out[out["version"].isin(["v11", "v11i"])].sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()

    md = ["# v11 — Yield Map Target-Clean Sensitivity\n\n"]
    md.append("Removed 8 `(field_id, year)` rows where yield map mean and factual target differ by >1 t/ha.\n\n")
    md.append("## Best overall\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Best among target-clean datasets\n\n")
    md.append(clean_best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(out.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "asof_v11_results.md").write_text("".join(md), encoding="utf-8")

    print("\nFINAL BEST")
    print(best.round(3).to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {REPORTS / 'asof_v11_results.md'}")


if __name__ == "__main__":
    main()
