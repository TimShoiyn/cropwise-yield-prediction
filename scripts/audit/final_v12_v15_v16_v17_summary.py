"""
Consolidate the v12 / v15 / v16 / v17 + Cropwise comparison.

Reads asof results CSVs from models_v2/asof_comparison/ and produces a single
report comparing best ML policy vs Cropwise on every (scenario, asof_tag).

Reports MAPE, MAE, RMSE, R^2 side-by-side. Also runs error analysis on the
best v17 policy (per scenario / date) to identify residual problem rows.

Outputs:
    reports/FINAL_V12_V15_V16_V17_COMPARISON.md
    reports/figures/final_metric_comparison.png
    reports/v17_error_analysis.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    AS_OF_LABELS,
    PARAM_SETS,
    REPORTS,
    attach_cropwise,
    cropwise_table,
    load_dataset,
    walk_forward_oof,
)

COMP_DIR = Path("models_v2/asof_comparison")
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["sunflower", "wheat_combined", "all_crops"]
ASOF_TAGS = ["07_01", "08_01", "09_01"]

CANDIDATE_FILES = {
    "v12": COMP_DIR / "asof_results_v12.csv",
    "v15": COMP_DIR / "asof_results_v15_policy.csv",
    "v16_clean": COMP_DIR / "asof_results_v16_conflict_clean.csv",
    "v17": COMP_DIR / "asof_results_v17_strict.csv",
}


def best_per_block(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    )


def collect_best() -> pd.DataFrame:
    rows = []
    for tag, path in CANDIDATE_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "version" not in df.columns:
            df["version"] = tag
        df["family"] = tag
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    best = best_per_block(full).copy()
    return best


def cropwise_row(best: pd.DataFrame) -> pd.DataFrame:
    cw = best[["scenario", "asof_tag", "asof_label", "n_cw", "cw_mape", "cw_rmse", "cw_mae", "cw_r2"]].copy()
    cw["family"] = "cropwise"
    cw = cw.rename(
        columns={
            "n_cw": "n",
            "cw_mape": "mape",
            "cw_rmse": "rmse",
            "cw_mae": "mae",
            "cw_r2": "r2",
        }
    )
    return cw


def _select_v17_choice(best_v17_row: pd.Series) -> tuple[str, str, str, bool]:
    model = best_v17_row["model"]
    version = best_v17_row["version"]
    if model.endswith("_default"):
        params = "default"
    elif model.endswith("_l30"):
        params = "shallow_l30"
    elif model.endswith("_l10"):
        params = "shallow_l10"
    elif model.endswith("_no_fid"):
        params = "shallow_l10"
    else:
        params = "shallow_l10"
    use_field_id = not model.endswith("_no_fid")
    return version, model, params, use_field_id


def error_analysis(best_v17: pd.DataFrame) -> pd.DataFrame:
    cw = cropwise_table()
    rows: list[pd.DataFrame] = []
    for _, r in best_v17.iterrows():
        scenario = r["scenario"]
        tag = r["asof_tag"]
        version, model, params_name, use_field_id = _select_v17_choice(r)
        crop_filter = (
            ["sunflower"] if scenario == "sunflower"
            else ["wheat_spring", "wheat_winter"] if scenario == "wheat_combined"
            else None
        )
        df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
        if df.empty:
            continue
        pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
        out = df.assign(pred=pred).dropna(subset=["pred"]).copy()
        out["abs_err_ml"] = (out["target_yield_t_ha"] - out["pred"]).abs()
        out["abs_err_cw"] = (out["target_yield_t_ha"] - out["cropwise_asof_t_ha"]).abs()
        out["scenario"] = scenario
        out["asof_tag"] = tag
        out["version_used"] = version
        out["model_used"] = model
        cols = [
            "scenario", "asof_tag", "version_used", "model_used",
            "field_id", "year", "standard_name",
            "target_yield_t_ha", "pred", "abs_err_ml",
            "cropwise_asof_t_ha", "abs_err_cw",
        ]
        rows.append(out[cols])
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(REPORTS / "v17_error_analysis.csv", index=False)
    return full


def plot_metric_comparison(best: pd.DataFrame, cw_df: pd.DataFrame, metric: str, ax) -> None:
    pivot_ml = best.pivot(index="asof_tag", columns="scenario", values=metric).reindex(ASOF_TAGS)
    pivot_cw = cw_df.pivot(index="asof_tag", columns="scenario", values=metric).reindex(ASOF_TAGS)
    pivot_ml = pivot_ml[SCENARIOS]
    pivot_cw = pivot_cw[SCENARIOS]
    x = np.arange(len(ASOF_TAGS))
    width = 0.12
    colors_ml = {"sunflower": "#FFB000", "wheat_combined": "#785EF0", "all_crops": "#1f77b4"}
    colors_cw = {"sunflower": "#FFE5B4", "wheat_combined": "#C7BBFF", "all_crops": "#aec7e8"}
    for i, sc in enumerate(SCENARIOS):
        ax.bar(x - width * 1.5 + i * width, pivot_ml[sc].values, width, label=f"ML/{sc}", color=colors_ml[sc])
        ax.bar(x + width * 0.5 + i * width, pivot_cw[sc].values, width, label=f"CW/{sc}", color=colors_cw[sc])
    ax.set_xticks(x)
    ax.set_xticklabels([AS_OF_LABELS[t] for t in ASOF_TAGS])
    ax.set_title(metric.upper())
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    best = collect_best()
    cw = cropwise_row(best)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, metric in zip(axes, ["mape", "mae", "rmse", "r2"]):
        plot_metric_comparison(best, cw, metric, ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_png = FIGURES / "final_metric_comparison.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    err = error_analysis(best[best["family"] == "v17"])

    md = ["# Final v12 / v15 / v16_clean / v17 + Cropwise comparison\n\n"]
    md.append("All numbers are walk-forward OOF; ML uses best policy per scenario / date.\n")
    md.append("Datasets:\n")
    md.append("- **v12** : prod_fact_t_ha target (ц/га → t/ha) for 30 fields with name match\n")
    md.append("- **v15** : v12 + v14 (curated management) + v15_rates (rates only) policy\n")
    md.append("- **v16_clean** : v15 minus 7 rows where productivity_data and yield_maps disagree by >1 t/ha\n")
    md.append("- **v17** : target rebuilt with priority physical_t_ha > prod_fact_t_ha; conflict rows dropped where physical and prod_fact disagree by >1 t/ha\n\n")
    md.append("## Best ML policy per scenario / date\n\n")
    md.append(best[
        ["scenario", "asof_tag", "asof_label", "family", "version", "model",
         "n_ml", "mape", "mae", "rmse", "r2", "cw_mape", "cw_mae", "cw_rmse", "cw_r2"]
    ].round(3).to_markdown(index=False))
    md.append("\n\n## Same view but ML vs Cropwise side-by-side\n\n")
    md.append("ML metrics are best-of-policy in column `family`/`model`; Cropwise metrics are computed on the same rows.\n\n")
    head = ["scenario", "asof_tag", "ML family", "ML model", "n", "ML MAPE", "CW MAPE", "ML MAE", "CW MAE", "ML RMSE", "CW RMSE", "ML R^2", "CW R^2"]
    rows = []
    for _, r in best.iterrows():
        rows.append([
            r["scenario"], AS_OF_LABELS[r["asof_tag"]], r["family"], r["model"],
            int(r["n_ml"]),
            f"{r['mape']:.1f}", f"{r['cw_mape']:.1f}",
            f"{r['mae']:.2f}", f"{r['cw_mae']:.2f}",
            f"{r['rmse']:.2f}", f"{r['cw_rmse']:.2f}",
            f"{r['r2']:.2f}", f"{r['cw_r2']:.2f}",
        ])
    table_df = pd.DataFrame(rows, columns=head)
    md.append(table_df.to_markdown(index=False))

    md.append("\n\n## v17 residual error: top 20 ML misses by abs error\n\n")
    if not err.empty:
        top = err.sort_values("abs_err_ml", ascending=False).head(20)
        md.append(top[
            ["scenario", "asof_tag", "field_id", "year", "standard_name",
             "target_yield_t_ha", "pred", "abs_err_ml", "cropwise_asof_t_ha", "abs_err_cw"]
        ].round(3).to_markdown(index=False))
        md.append("\n\n## v17 residual error: top 20 ML misses where Cropwise was right (CW err < ML err - 1)\n\n")
        better_cw = err[err["abs_err_cw"] + 1.0 < err["abs_err_ml"]].sort_values("abs_err_ml", ascending=False).head(20)
        if not better_cw.empty:
            md.append(better_cw[
                ["scenario", "asof_tag", "field_id", "year", "standard_name",
                 "target_yield_t_ha", "pred", "abs_err_ml", "cropwise_asof_t_ha", "abs_err_cw"]
            ].round(3).to_markdown(index=False))
        else:
            md.append("No such rows: ML never lost to CW by more than 1 t/ha.\n")

    md.append("\n\n## Notes on metrics\n\n")
    md.append(
        "- **MAPE** normalizes per-row error by `y_true`, so it is comparable across crops with different yield "
        "scales. We report it as the headline metric because Cropwise itself reports percentage errors.\n"
    )
    md.append(
        "- **R^2** is fragile on small per-year folds; one outlier can flip its sign. ML R^2 stays near 0 because the "
        "model effectively predicts close to the train-year mean; Cropwise has a higher R^2 (it ranks fields better) "
        "but a worse MAPE (it absolute-errors more when yield is far from average).\n"
    )
    md.append(
        "- **MAE / RMSE** stay in t/ha units; Cropwise wins on `wheat_combined` and `all_crops` even when ML wins on MAPE.\n"
    )

    md.append(f"\n![Metric comparison]({out_png.relative_to(REPORTS)})\n")

    out = REPORTS / "FINAL_V12_V15_V16_V17_COMPARISON.md"
    out.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {out_png}")
    print("\nFinal best table:")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
