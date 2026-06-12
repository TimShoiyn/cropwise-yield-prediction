"""
Generate minimal supervisor artifacts for the "phases" experiment (no new experiments).

Inputs (already produced by train_phases_experiment.py):
  - models/models_phases_comparison.csv
  - models/oof_phases_baseline_allcrops.csv
  - models/oof_phases_phases_allcrops.csv
  - data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv

Outputs:
  - models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png
  - models/plots/phases_cv_r2_barplot_3subsets.png
  - models/plots/phases_feature_importance_allcrops_phases_top20.png
  - models/models_phases_summary_short.csv
  - models/phases_interpretation_short.txt
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_baseline_model import (
    BEST_REG_GBDT_PARAMS,
    build_pipeline_with_params,
    evaluate_with_oof,
    prepare_features,
)
from sklearn.model_selection import GroupKFold


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

MODELS_DIR = Path("models")
PLOTS_DIR = MODELS_DIR / "plots"
DATA_PROCESSED_DIR = Path("data_processed")

COMPARISON_CSV = MODELS_DIR / "models_phases_comparison.csv"
OOF_BASE_ALL = MODELS_DIR / "oof_phases_baseline_allcrops.csv"
OOF_PHASES_ALL = MODELS_DIR / "oof_phases_phases_allcrops.csv"
PHASES_DATASET = DATA_PROCESSED_DIR / "ml_dataset_full_extended_cropwindow_v1_phases.csv"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _scatter_allcrops() -> Path:
    _require(OOF_BASE_ALL)
    _require(OOF_PHASES_ALL)
    df_b = pd.read_csv(OOF_BASE_ALL)
    df_p = pd.read_csv(OOF_PHASES_ALL)

    for df in (df_b, df_p):
        if not {"target_yield_t_ha", "model_pred_oof_t_ha"}.issubset(df.columns):
            raise RuntimeError("OOF file missing required columns target_yield_t_ha/model_pred_oof_t_ha")

    y_min = float(np.nanmin([df_b["target_yield_t_ha"].min(), df_p["target_yield_t_ha"].min()]))
    y_max = float(np.nanmax([df_b["target_yield_t_ha"].max(), df_p["target_yield_t_ha"].max()]))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phases_oof_scatter_allcrops_baseline_vs_phases.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    axes[0].scatter(df_b["target_yield_t_ha"], df_b["model_pred_oof_t_ha"], alpha=0.55, s=18)
    axes[0].plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
    axes[0].set_title("OOF: baseline (allcrops)")
    axes[0].set_xlabel("Actual yield (t/ha)")
    axes[0].set_ylabel("Predicted yield (t/ha)")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(df_p["target_yield_t_ha"], df_p["model_pred_oof_t_ha"], alpha=0.55, s=18)
    axes[1].plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
    axes[1].set_title("OOF: phases (allcrops)")
    axes[1].set_xlabel("Actual yield (t/ha)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Predicted vs Actual (OOF) - allcrops", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    return out


def _barplot_cv_r2() -> Path:
    _require(COMPARISON_CSV)
    comp = pd.read_csv(COMPARISON_CSV)

    need = {"subset_tag", "dataset_tag", "cv_r2_mean", "cv_r2_std"}
    if not need.issubset(comp.columns):
        raise RuntimeError(f"models_phases_comparison.csv missing columns: {sorted(need - set(comp.columns))}")

    # keep consistent order
    subset_order = ["allcrops", "sunflower", "wheat"]
    comp = comp[comp["subset_tag"].isin(subset_order) & comp["dataset_tag"].isin(["baseline", "phases"])].copy()
    comp["subset_tag"] = pd.Categorical(comp["subset_tag"], categories=subset_order, ordered=True)

    # Plot grouped bars with errorbars (std)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phases_cv_r2_barplot_3subsets.png"

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(
        data=comp,
        x="subset_tag",
        y="cv_r2_mean",
        hue="dataset_tag",
        ax=ax,
        palette={"baseline": "#4C72B0", "phases": "#55A868"},
    )

    # add errorbars manually
    for i, row in comp.sort_values(["subset_tag", "dataset_tag"]).reset_index(drop=True).iterrows():
        x = i // 2  # two bars per group (baseline, phases)
        # bar positions from seaborn are tricky; use patches order
    for patch, (_, row) in zip(ax.patches, comp.sort_values(["subset_tag", "dataset_tag"]).iterrows()):
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        err = float(row["cv_r2_std"]) if pd.notna(row["cv_r2_std"]) else 0.0
        ax.errorbar(x, y, yerr=err, color="black", capsize=4, lw=1)

    ax.axhline(0.0, color="black", lw=1, alpha=0.6)
    ax.set_title("CV R2 (GroupKFold by year): baseline vs phases", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("CV R2 mean (higher is better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    return out


def _feature_importance_allcrops_phases() -> Path:
    """
    Compute Top-20 feature importance for the 'phases' allcrops model.

    Note: this fits the same model once on the full phases dataset (in-sample),
    strictly for interpretation/importance visualization.
    """
    _require(PHASES_DATASET)
    df = pd.read_csv(PHASES_DATASET)

    X, y, groups, feature_cols = prepare_features(df, model_type="all", exclude_leaky=True)

    pipe = build_pipeline_with_params(
        feature_cols,
        n_estimators=BEST_REG_GBDT_PARAMS["n_estimators"],
        max_depth=BEST_REG_GBDT_PARAMS["max_depth"],
        learning_rate=BEST_REG_GBDT_PARAMS["learning_rate"],
        min_samples_leaf=BEST_REG_GBDT_PARAMS["min_samples_leaf"],
        subsample=BEST_REG_GBDT_PARAMS["subsample"],
    )

    # Fit on full data
    pipe.fit(X, y.to_numpy(dtype=float))

    # Extract feature names after preprocessing
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
        feat_names = [str(x) for x in feat_names]
    except Exception:
        feat_names = [str(c) for c in X.columns]

    importances = np.asarray(pipe.named_steps["model"].feature_importances_, dtype=float)
    imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    top = imp.head(20).iloc[::-1]  # reverse for horizontal barplot (top at top)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phases_feature_importance_allcrops_phases_top20.png"

    plt.figure(figsize=(10.5, 7.5))
    sns.barplot(data=top, x="importance", y="feature", palette="viridis")
    plt.title("Top-20 feature importance (phases model, allcrops)", fontsize=13, fontweight="bold")
    plt.xlabel("GBDT feature importance (impurity-based)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    # also save the raw table for convenience (optional, tiny)
    imp.to_csv(MODELS_DIR / "phases_feature_importance_allcrops_full.csv", index=False)

    return out


def _write_short_summary_and_text() -> tuple[Path, Path]:
    _require(COMPARISON_CSV)
    comp = pd.read_csv(COMPARISON_CSV)

    # Short metrics table
    keep = comp[comp["dataset_tag"].isin(["baseline", "phases"])].copy()
    keep = keep[keep["subset_tag"].isin(["allcrops", "sunflower", "wheat"])].copy()
    short = keep[[
        "subset_tag",
        "dataset_tag",
        "cv_r2_mean",
        "cv_rmse_mean",
        "cv_mape_mean",
        "cv_mae_mean",
        "n_rows",
        "n_features",
    ]].copy()
    short = short.sort_values(["subset_tag", "dataset_tag"])

    out_csv = MODELS_DIR / "models_phases_summary_short.csv"
    short.to_csv(out_csv, index=False)

    # Interpretation text (5-7 sentences)
    def _row(sub: str, tag: str) -> pd.Series:
        r = comp[(comp["subset_tag"] == sub) & (comp["dataset_tag"] == tag)]
        if len(r) != 1:
            return pd.Series(dtype=float)
        return r.iloc[0]

    a_b = _row("allcrops", "baseline")
    a_p = _row("allcrops", "phases")
    s_b = _row("sunflower", "baseline")
    s_p = _row("sunflower", "phases")
    w_b = _row("wheat", "baseline")
    w_p = _row("wheat", "phases")

    lines = []
    lines.append(
        "Experiment summary (GroupKFold by year, OOF metrics)."
    )
    lines.append(
        f"All crops: phase-based features improved CV R2 from {a_b.get('cv_r2_mean', np.nan):.3f} to {a_p.get('cv_r2_mean', np.nan):.3f} "
        f"and reduced CV RMSE from {a_b.get('cv_rmse_mean', np.nan):.2f} to {a_p.get('cv_rmse_mean', np.nan):.2f} t/ha "
        f"(MAPE {a_b.get('cv_mape_mean', np.nan):.1f}% -> {a_p.get('cv_mape_mean', np.nan):.1f}%)."
    )
    lines.append(
        "This suggests that aligning NDVI and weather information to phenological time (GDD phases) adds useful signal at the mixed-crops level."
    )
    lines.append(
        f"Sunflower-only: metrics got slightly worse (CV R2 {s_b.get('cv_r2_mean', np.nan):.3f} -> {s_p.get('cv_r2_mean', np.nan):.3f}), "
        "which is plausible because the current phase feature set for sunflower is limited (mainly hot-days by phase) and may add noise without enough supporting weather/NDVI phase summaries."
    )
    lines.append(
        f"Wheat-only (N={int(w_b.get('n_rows', 0))}): CV R2 is unstable and negative in both cases ({w_b.get('cv_r2_mean', np.nan):.3f} -> {w_p.get('cv_r2_mean', np.nan):.3f}); "
        "with small sample sizes per year, leaving one year out can dominate results and make added features appear harmful."
    )
    lines.append(
        "For the supervisor: we should treat the allcrops improvement as the main positive result, and interpret crop-specific degradation as a sign that crop-specific phase features need refinement and/or more data per crop-year before they reliably help."
    )

    out_txt = MODELS_DIR / "phases_interpretation_short.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_csv, out_txt


def main() -> None:
    sns.set_style("whitegrid")

    p1 = _scatter_allcrops()
    p2 = _barplot_cv_r2()
    p3 = _feature_importance_allcrops_phases()
    out_csv, out_txt = _write_short_summary_and_text()

    print("Done. Created:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"  {out_csv}")
    print(f"  {out_txt}")


if __name__ == "__main__":
    main()

