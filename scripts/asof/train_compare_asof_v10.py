"""
v10 experiment: v9 + internal-only scout/soil features.

Compares v7, best v9-style params, v10, v10 without field_id, and Cropwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from scripts.asof.train_compare_asof_v9 import (
    AS_OF_LABELS,
    AS_OF_TAGS,
    COMP_DIR,
    FIGS,
    REPORTS,
    PARAM_SETS,
    attach_cropwise,
    cropwise_table,
    load_dataset,
    metrics,
    walk_forward_oof,
)


def evaluate_model(version: str, model_name: str, crop_filter, tag: str, cw_table_df, params_name: str, use_field_id: bool = True):
    df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw_table_df)
    if len(df) < 30:
        return None
    pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
    cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
    if cmp_df.empty:
        return None
    cmp_cw = cmp_df.dropna(subset=["cropwise_asof_t_ha"])
    ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
    cw = metrics(cmp_cw["target_yield_t_ha"], cmp_cw["cropwise_asof_t_ha"]) if len(cmp_cw) else {}
    return {
        "model": model_name,
        "n_ml": len(cmp_df),
        "n_cw": len(cmp_cw),
        "mape": ml["mape"],
        "rmse": ml["rmse"],
        "mae": ml["mae"],
        "r2": ml["r2"],
        "cw_mape": cw.get("mape", float("nan")),
        "cw_rmse": cw.get("rmse", float("nan")),
        "cw_mae": cw.get("mae", float("nan")),
        "cw_r2": cw.get("r2", float("nan")),
    }


def run_one(scenario: str, crop_filter, tag: str, cw_table_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = [
        ("v7", "v7_default", "default", True),
        ("v9", "v9_shallow_l10", "shallow_l10", True),
        ("v9", "v9_mae_l30", "mae_l30", True),
        ("v9", "v9_no_field_id", "shallow_l10", False),
        ("v10", "v10_default", "default", True),
        ("v10", "v10_shallow_l10", "shallow_l10", True),
        ("v10", "v10_shallow_l30", "shallow_l30", True),
        ("v10", "v10_mae_l30", "mae_l30", True),
        ("v10", "v10_no_field_id", "shallow_l10", False),
    ]
    for version, model_name, params_name, use_field_id in candidates:
        result = evaluate_model(version, model_name, crop_filter, tag, cw_table_df, params_name, use_field_id)
        if result is None:
            continue
        result.update({
            "scenario": scenario,
            "asof_tag": tag,
            "asof_label": AS_OF_LABELS[tag],
        })
        rows.append(result)
    return pd.DataFrame(rows)


def plot_best(summary: pd.DataFrame) -> Path:
    best = summary.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    plt.figure(figsize=(9, 5.5))
    for scenario in best["scenario"].unique():
        s = best[best["scenario"] == scenario].sort_values("asof_tag")
        plt.plot(s["asof_label"], s["mape"], marker="o", lw=2.5, label=f"best ML - {scenario}")
        plt.plot(s["asof_label"], s["cw_mape"], marker="s", lw=2, ls="--", color="grey", label=f"Cropwise - {scenario}")
    plt.title("v10 internal-only features: best ML vs Cropwise")
    plt.xlabel("As-of forecast date")
    plt.ylabel("MAPE, %")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = FIGS / "asof_mape_v10_best.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def main() -> None:
    print("=" * 80)
    print("TRAIN v10 — INTERNAL SCOUT + SOIL FEATURES")
    print("=" * 80)
    cw = cropwise_table()
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    parts = []
    for scenario, crop_filter in scenarios:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            result = run_one(scenario, crop_filter, tag, cw)
            if result.empty:
                print(f"  {tag}: no result")
                continue
            best = result.sort_values("mape").iloc[0]
            v9 = result[result["model"].str.startswith("v9")].sort_values("mape").head(1)
            v10 = result[result["model"].str.startswith("v10")].sort_values("mape").head(1)
            v9_txt = f"v9best={v9.iloc[0]['mape']:.1f}%" if not v9.empty else "v9best=nan"
            v10_txt = f"v10best={v10.iloc[0]['mape']:.1f}%" if not v10.empty else "v10best=nan"
            print(f"  {tag}: best={best['model']} MAPE={best['mape']:.1f}% R2={best['r2']:.3f}; {v9_txt}; {v10_txt}; CW={best['cw_mape']:.1f}%")
            parts.append(result)
    if not parts:
        print("No results")
        return

    summary = pd.concat(parts, ignore_index=True)
    out_csv = COMP_DIR / "asof_results_v10.csv"
    summary.to_csv(out_csv, index=False)
    best = summary.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    plot_path = plot_best(summary)

    md = ["# v10 — Internal Scout + Soil Features\n\n"]
    md.append("_Generated by `scripts/asof/train_compare_asof_v10.py`._\n\n")
    md.append("No API data, no Cropwise forecast feature. Added only as-of scout reports and soil samples.\n\n")
    md.append("## Best model per scenario/date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    md.append(f"![v10 best](figures/{plot_path.name})\n")
    (REPORTS / "asof_v10_results.md").write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL BEST")
    print("=" * 80)
    print(best.round(3).to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {REPORTS / 'asof_v10_results.md'}")


if __name__ == "__main__":
    main()
