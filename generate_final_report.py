"""
Generate final ML report for Cropwise API ML pipeline.

Usage:
    python generate_final_report.py

Outputs (under reports/):
    - tables/models_comparison_summary.{csv,png}
    - tables/dataset_statistics.{csv,png}
    - figures/cv_r2_comparison.png
    - figures/feature_importance_best_model.png
    - figures/predictions_vs_actual_best_model.png
    - final_report.pdf (optional, if PdfPages is available)
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


# Paths
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data_processed"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create reports/ subfolders if needed."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Dataset metadata (same naming as in train_baseline_model.py)
DATASET_PATHS = {
    "base_with_ndvi": DATA_DIR / "ml_dataset_with_ndvi.csv",
    "ndvi_only_extended": DATA_DIR / "ml_dataset_ndvi_only_extended.csv",
    "full_extended": DATA_DIR / "ml_dataset_full_extended.csv",
    "ops_ndvi_2021_2025": DATA_DIR / "ml_dataset_ops_ndvi_2021_2025.csv",
}


def load_n_obs_per_dataset() -> dict:
    """Return mapping dataset_name -> number of observations (rows)."""
    result: dict[str, int] = {}
    for name, path in DATASET_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            result[name] = len(df)
        else:
            result[name] = np.nan
    return result


def build_models_comparison_summary() -> pd.DataFrame:
    """
    Build nice summary table from models/models_comparison.csv.

    - Filter target datasets.
    - Keep only key models per dataset.
    - Add N obs column.
    """
    comparison_path = MODELS_DIR / "models_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"{comparison_path} not found")

    df = pd.read_csv(comparison_path)

    target_datasets = [
        "base_with_ndvi",
        "ndvi_only_extended",
        "full_extended",
        "ops_ndvi_2021_2025",
    ]
    df = df[df["dataset_name"].isin(target_datasets)].copy()

    # Keep only selected model types
    keep_models = {
        "no_ndvi_no_leak",
        "ndvi_only",
        "all",
        "all_no_leak",
    }
    df = df[df["model_type"].isin(keep_models)].copy()

    # Map to nicer labels
    model_label_map = {
        "no_ndvi_no_leak": "no_ndvi_no_leak",
        "no_ndvi": "no_ndvi",
        "ndvi_only": "ndvi_only",
        "all_no_leak": "all_no_leak",
        "all": "all",
    }
    df["Model"] = df["model_type"].map(model_label_map).fillna(df["model_type"])
    df["Dataset"] = df["dataset_name"]

    n_obs_map = load_n_obs_per_dataset()
    df["N obs"] = df["dataset_name"].map(n_obs_map)

    # Select & rename columns
    out = df[
        [
            "Dataset",
            "Model",
            "N obs",
            "cv_r2_mean",
            "cv_rmse_mean",
            "cv_mae_mean",
            "cv_mape_mean",
            "train_r2",
            "train_rmse",
        ]
    ].copy()

    out = out.rename(
        columns={
            "cv_r2_mean": "CV R2",
            "cv_rmse_mean": "CV RMSE (t/ha)",
            "cv_mae_mean": "CV MAE (t/ha)",
            "cv_mape_mean": "CV MAPE (%)",
            "train_r2": "Train R2",
            "train_rmse": "Train RMSE (t/ha)",
        }
    )

    # Rounding
    out["CV R2"] = out["CV R2"].round(3)
    out["CV RMSE (t/ha)"] = out["CV RMSE (t/ha)"].round(2)
    out["CV MAE (t/ha)"] = out["CV MAE (t/ha)"].round(2)
    out["CV MAPE (%)"] = out["CV MAPE (%)"].round(1)
    out["Train R2"] = out["Train R2"].round(3)
    out["Train RMSE (t/ha)"] = out["Train RMSE (t/ha)"].round(2)

    # Save CSV
    csv_path = TABLES_DIR / "models_comparison_summary.csv"
    out.to_csv(csv_path, index=False)
    print(f"[OK] Saved models comparison summary CSV -> {csv_path}")

    # Save PNG table
    png_path = TABLES_DIR / "models_comparison_summary.png"
    save_table_png(
        out,
        png_path,
        title="Models comparison summary",
        highlight_max_col="CV R2",
        group_by="Dataset",
    )
    print(f"[OK] Saved models comparison summary PNG -> {png_path}")

    return out


def save_table_png(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    highlight_max_col: str | None = None,
    group_by: str | None = None,
) -> None:
    """
    Render a pandas DataFrame as a PNG table via matplotlib.

    - highlight_max_col/group_by: highlight max value in this column per group.
    """
    # Prepare cell colours (same size as df)
    cell_colours = [["white"] * len(df.columns) for _ in range(len(df))]

    if highlight_max_col is not None and highlight_max_col in df.columns:
        if group_by is not None and group_by in df.columns:
            for dataset, grp in df.groupby(group_by):
                # idx of row with max value in this group
                idx = grp[highlight_max_col].idxmax()
                row_pos = df.index.get_loc(idx)
                col_pos = df.columns.get_loc(highlight_max_col)
                cell_colours[row_pos][col_pos] = "#d0f0c0"  # light green
        else:
            idx = df[highlight_max_col].idxmax()
            row_pos = df.index.get_loc(idx)
            col_pos = df.columns.get_loc(highlight_max_col)
            cell_colours[row_pos][col_pos] = "#d0f0c0"

    fig_height = max(2, 0.5 * len(df) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colours,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_cv_r2_comparison(summary_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing CV R² by dataset and model."""
    df_plot = summary_df[["Dataset", "Model", "CV R2"]].copy()

    # Order for plotting
    dataset_order = [
        "base_with_ndvi",
        "ndvi_only_extended",
        "full_extended",
        "ops_ndvi_2021_2025",
    ]
    model_order = ["no_ndvi_no_leak", "ndvi_only", "all_no_leak", "all"]

    df_plot = df_plot[df_plot["Dataset"].isin(dataset_order)]
    df_plot["Dataset"] = pd.Categorical(df_plot["Dataset"], categories=dataset_order, ordered=True)
    df_plot["Model"] = pd.Categorical(df_plot["Model"], categories=model_order, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(
        data=df_plot,
        x="Dataset",
        y="CV R2",
        hue="Model",
        palette="muted",
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Comparison of CV R² across Datasets and Models", fontsize=14, fontweight="bold")
    plt.ylabel("CV R²")
    plt.xlabel("Dataset")
    plt.tight_layout()

    out_path = FIGURES_DIR / "cv_r2_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved CV R2 comparison figure -> {out_path}")


def copy_existing_figures() -> None:
    """
    Copy existing best-model figures into reports/figures.

    If the source files don't exist (e.g. user changed training config),
    this function will just skip them.
    """
    # Feature importance for best model
    src_fi = MODELS_DIR / "feature_importance_full_extended_all_no_leak.png"
    dst_fi = FIGURES_DIR / "feature_importance_best_model.png"
    if src_fi.exists():
        shutil.copyfile(src_fi, dst_fi)
        print(f"[OK] Copied feature importance figure -> {dst_fi}")
    else:
        print(f"[WARN] {src_fi} not found, skipping copy")

    # Predictions vs actual for best model
    # После Iteration 1 сохраняется OOF-версия графика
    src_pred = MODELS_DIR / "predictions_plot_full_extended_all_no_leak_oof.png"
    dst_pred = FIGURES_DIR / "predictions_vs_actual_best_model.png"
    if src_pred.exists():
        shutil.copyfile(src_pred, dst_pred)
        print(f"[OK] Copied predictions vs actual figure -> {dst_pred}")
    else:
        # fallback на старое имя, если осталось от предыдущего прогона
        old_src = MODELS_DIR / "predictions_plot_full_extended_all_no_leak.png"
        if old_src.exists():
            shutil.copyfile(old_src, dst_pred)
            print(f"[OK] Copied predictions vs actual figure (fallback) -> {dst_pred}")
        else:
            print(f"[WARN] {src_pred} not found, skipping copy")


def build_dataset_statistics() -> pd.DataFrame:
    """Compute simple stats for each dataset."""
    rows: list[dict] = []
    for name, path in DATASET_PATHS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)

        n_obs = len(df)
        n_fields = df["field_id"].nunique() if "field_id" in df.columns else np.nan
        year_min = int(df["year"].min()) if "year" in df.columns else np.nan
        year_max = int(df["year"].max()) if "year" in df.columns else np.nan
        years_range = f"{year_min}–{year_max}" if not (pd.isna(year_min) or pd.isna(year_max)) else ""

        if "target_yield_t_ha" in df.columns:
            mean_y = float(df["target_yield_t_ha"].mean())
            std_y = float(df["target_yield_t_ha"].std())
        else:
            mean_y = np.nan
            std_y = np.nan

        # Number of features: all columns minus service ones
        service_cols = {"field_id", "year", "target_yield_t_ha"}
        n_features = len([c for c in df.columns if c not in service_cols])

        rows.append(
            {
                "Dataset": name,
                "N obs": n_obs,
                "N fields": n_fields,
                "Years": years_range,
                "Mean yield (t/ha)": round(mean_y, 2) if not pd.isna(mean_y) else np.nan,
                "Std yield (t/ha)": round(std_y, 2) if not pd.isna(std_y) else np.nan,
                "N features": n_features,
            }
        )

    stats_df = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "dataset_statistics.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved dataset statistics CSV -> {csv_path}")

    png_path = TABLES_DIR / "dataset_statistics.png"
    save_table_png(stats_df, png_path, title="Dataset statistics")
    print(f"[OK] Saved dataset statistics PNG -> {png_path}")

    return stats_df


def build_final_pdf() -> None:
    """
    Assemble a multi-page PDF from generated PNGs.

    If something is missing, pages are skipped.
    """
    pdf_path = REPORTS_DIR / "final_report.pdf"
    figures = [
        TABLES_DIR / "models_comparison_summary.png",
        FIGURES_DIR / "cv_r2_comparison.png",
        FIGURES_DIR / "feature_importance_best_model.png",
        FIGURES_DIR / "predictions_vs_actual_best_model.png",
        TABLES_DIR / "dataset_statistics.png",
    ]

    with PdfPages(pdf_path) as pdf:
        for fig_path in figures:
            if not fig_path.exists():
                print(f"[WARN] Figure not found, skipping in PDF: {fig_path}")
                continue
            img = plt.imread(fig_path)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Saved final PDF report -> {pdf_path}")


def main() -> None:
    ensure_dirs()

    print("=== Building models comparison summary ===")
    summary_df = build_models_comparison_summary()

    print("=== Building CV R2 comparison figure ===")
    build_cv_r2_comparison(summary_df)

    print("=== Copying existing best-model figures ===")
    copy_existing_figures()

    print("=== Building dataset statistics ===")
    build_dataset_statistics()

    print("=== Building final PDF report ===")
    build_final_pdf()

    print("=== DONE: reports generated in ./reports ===")


if __name__ == "__main__":
    main()

