"""v30 policy model summary.

v29 showed that the best feature branch is crop/scenario-specific, not universal:

  - all_crops / wheat_combined: full management (`pred_v28_all_mgmt`)
  - sunflower: pruned management (`pred_v29_pruned_mgmt`)
  - barley: no management (`pred_v27_s2`)

This script materializes the selected policy predictions for the main date
(`08_01`) and writes a concise final report.
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

COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
MAIN_TAG = "08_01"

POLICY = {
    "all_crops": ("pred_v28_all_mgmt", "v28_all_mgmt"),
    "wheat_combined": ("pred_v28_all_mgmt", "v28_all_mgmt"),
    "sunflower": ("pred_v29_pruned_mgmt", "v29_pruned_mgmt"),
    "barley": ("pred_v27_s2", "v27_s2"),
}


def block(df: pd.DataFrame, scenario: str, model: str, pred_col: str) -> dict[str, object]:
    sub = df.dropna(subset=[pred_col]).copy()
    m = ev.metrics(sub["target_yield_t_ha"], sub[pred_col])
    return {
        "scenario": scenario,
        "asof_tag": MAIN_TAG,
        "asof": "1 Aug",
        "model": model,
        "n": len(sub),
        "mape": m["mape"],
        "r2": m["r2"],
        "wcy_r2": ev.within_crop_year_r2(sub, pred_col),
        "rmse": m["rmse"],
        "mae": m["mae"],
    }


def main() -> None:
    preds = pd.read_csv(COMP_DIR / "asof_predictions_v29_pruned_management.csv")
    preds = preds[preds["asof_tag"].eq(MAIN_TAG)].copy()

    rows = []
    out_rows = []
    for scenario, (pred_col, selected_model) in POLICY.items():
        df = preds[preds["scenario"].eq(scenario)].copy()
        if df.empty:
            continue
        df["pred_v30_policy"] = df[pred_col]
        df["v30_selected_model"] = selected_model
        fair = df[df["pred_v30_policy"].notna()].copy()
        rows.append(block(fair, scenario, "v30_policy", "pred_v30_policy"))
        rows.append(block(fair, scenario, "cropwise_asof", "cropwise_asof_t_ha"))
        out_rows.append(
            df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_v30_policy",
                    "v30_selected_model",
                    "scenario",
                    "asof_tag",
                ]
            ]
        )

    res = pd.DataFrame(rows)
    selected = pd.concat(out_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v30_policy.csv", index=False)
    selected.to_csv(COMP_DIR / "asof_predictions_v30_policy.csv", index=False)

    show = res.copy()
    for col in ["mape", "r2", "wcy_r2", "rmse", "mae"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))

    md = ["# v30 — финальная policy-модель (1 Aug)\n\n"]
    md.append("Дата: 2026-06-11.\n\n")
    md.append("Основная дата: **1 августа** (лучший компромисс: до уборки, сезонный сигнал уже есть).\n\n")
    md.append("Policy выбирает лучшую ветку по сценарию/культуре:\n\n")
    md.append("| Сценарий | Выбранная ветка |\n|---|---|\n")
    for scenario, (_, selected_model) in POLICY.items():
        md.append(f"| `{scenario}` | `{selected_model}` |\n")
    md.append("\n## Метрики\n\n")
    md.append(show[["scenario", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n\n")
    md.append("## Короткий вывод\n\n")
    md.append(
        "- `sunflower`: наша policy-модель лучше Cropwise-asof по MAPE и R²; это самый сильный результат.\n"
    )
    md.append(
        "- `wheat/all_crops`: полный management помогает, но Cropwise-asof всё ещё сильнее.\n"
    )
    md.append(
        "- `barley`: management шумит; лучшая наша ветка без management почти равна Cropwise по MAPE, но хуже по R².\n"
    )
    md.append(
        "- Универсальной одной модели нет: лучший результат получается через crop-specific policy.\n"
    )
    (REPORTS / "V30_POLICY_FINAL_RU.md").write_text("".join(md), encoding="utf-8")

    print(show[["scenario", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_string(index=False))
    print("\nWrote v30 policy results + report.")


if __name__ == "__main__":
    main()
