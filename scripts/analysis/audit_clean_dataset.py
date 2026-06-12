"""
Sprint 1.3 — EDA audit of the clean dataset v2.

Generates:
  - reports/figures/target_boxplot_per_crop.png
  - reports/figures/target_hist_per_crop_top4.png
  - reports/figures/target_year_trend_per_crop.png
  - reports/figures/nan_ratio_per_feature.png
  - reports/figures/feature_target_corr_per_crop.png
  - reports/clean_dataset_audit.md
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DATASET = DATA_PROCESSED / "ml_dataset_clean_v2.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(
        columns={"id": "crop_id"}
    )
    df = df.merge(crops, on="crop_id", how="left")
    return df


def fig_target_boxplot(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(11, 5))
    order = (
        df.groupby("standard_name")["target_yield_t_ha"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(data=df, x="standard_name", y="target_yield_t_ha", order=order)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Yield, т/га (cleaned)")
    plt.xlabel("Crop")
    plt.title("Target distribution per crop (after unit normalization)")
    plt.tight_layout()
    p = FIGS / "target_boxplot_per_crop.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def fig_target_hist_top4(df: pd.DataFrame) -> Path:
    top = (
        df["standard_name"].value_counts().head(4).index.tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, crop in zip(axes.ravel(), top):
        sub = df[df["standard_name"] == crop]["target_yield_t_ha"].dropna()
        ax.hist(sub, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(sub.median(), color="red", linestyle="--", label=f"median={sub.median():.2f}")
        ax.set_title(f"{crop}  (n={len(sub)})")
        ax.set_xlabel("Yield, т/га")
        ax.set_ylabel("count")
        ax.legend()
    plt.tight_layout()
    p = FIGS / "target_hist_per_crop_top4.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def fig_target_year_trend(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(11, 6))
    top = df["standard_name"].value_counts().head(4).index.tolist()
    sub = df[df["standard_name"].isin(top)].copy()
    g = (
        sub.groupby(["standard_name", "year"])["target_yield_t_ha"]
        .median()
        .reset_index()
    )
    sns.lineplot(data=g, x="year", y="target_yield_t_ha", hue="standard_name", marker="o")
    plt.title("Median yield per year per crop (top-4 crops)")
    plt.ylabel("Yield, т/га")
    plt.tight_layout()
    p = FIGS / "target_year_trend_per_crop.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def fig_nan_ratio(df: pd.DataFrame) -> Path:
    nans = (df.isna().sum() / len(df) * 100).sort_values(ascending=True)
    nans = nans[nans > 0]
    plt.figure(figsize=(9, max(4, len(nans) * 0.25)))
    nans.plot(kind="barh", color="indianred")
    plt.xlabel("NaN %")
    plt.title("NaN ratio per feature (clean v2)")
    plt.tight_layout()
    p = FIGS / "nan_ratio_per_feature.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def fig_corr_per_crop(df: pd.DataFrame) -> Path:
    """Heatmap: top NDVI/weather/soil features × target, per dominant crop."""
    crops = ["sunflower", "wheat_spring", "wheat_winter", "barley_spring"]
    feats = [
        "ndvi_mean_season",
        "ndvi_max_season",
        "ndvi_early",
        "ndvi_mid",
        "ndvi_late",
        "ndvi_observations",
        "weather_temp_avg_season",
        "weather_gdd_season",
        "weather_precip_sum_season",
        "weather_precip_sum_early",
        "weather_hot_days",
        "field_soil_pH",
        "field_soil_OM",
        "field_soil_P",
        "field_soil_K",
        "field_soil_N",
        "field_soil_Mg",
        "field_tillable_area",
    ]
    feats = [f for f in feats if f in df.columns]
    corr_mat = pd.DataFrame(index=feats, columns=crops, dtype=float)
    for c in crops:
        sub = df[df["standard_name"] == c]
        if len(sub) < 5:
            continue
        for f in feats:
            v = sub[[f, "target_yield_t_ha"]].dropna()
            if len(v) < 5:
                corr_mat.loc[f, c] = np.nan
                continue
            corr_mat.loc[f, c] = v.corr().iloc[0, 1]
    plt.figure(figsize=(8, 9))
    sns.heatmap(
        corr_mat.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title("Feature × target correlation per crop")
    plt.tight_layout()
    p = FIGS / "feature_target_corr_per_crop.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def write_md(df: pd.DataFrame, paths: dict[str, Path]) -> Path:
    md = []
    md.append("# Clean Dataset v2 — EDA audit\n\n")
    md.append("_Generated by `scripts/analysis/audit_clean_dataset.py`_\n\n")
    md.append(f"## Shape\n\n- Rows: **{len(df)}**\n- Columns: **{len(df.columns)}**\n\n")

    md.append("## Target distribution per crop (т/га, cleaned)\n\n")
    summary = (
        df.groupby("standard_name")
        .agg(
            n=("target_yield_t_ha", "size"),
            t_min=("target_yield_t_ha", "min"),
            t_med=("target_yield_t_ha", "median"),
            t_mean=("target_yield_t_ha", "mean"),
            t_max=("target_yield_t_ha", "max"),
            t_std=("target_yield_t_ha", "std"),
        )
        .sort_values("n", ascending=False)
    )
    md.append(summary.round(3).to_markdown())
    md.append("\n\n")

    md.append("## Year × crop coverage\n\n")
    pivot = (
        df.groupby(["year", "standard_name"]).size().unstack(fill_value=0)
    )
    md.append(pivot.to_markdown())
    md.append("\n\n")

    md.append("## NaN ratio per feature\n\n")
    nans = (df.isna().sum() / len(df) * 100).round(1).sort_values(ascending=False)
    nans = nans[nans > 0]
    md.append(nans.to_frame("nan_pct").to_markdown())
    md.append("\n\n")

    md.append("## Visualizations\n\n")
    for k, p in paths.items():
        md.append(f"### {k}\n\n![{k}]({p.relative_to(REPORTS).as_posix()})\n\n")

    md.append("## Notes for next sprint\n\n")
    md.append("- Sunflower (n=286) and wheat_combined (n=106) are the only crops with enough data for proper per-crop modeling.\n")
    md.append("- Soil features are essentially constants per field (one measurement per field, copied across years), so they encode 'inherent field fertility', not year-specific signal.\n")
    md.append("- Phase NDVI/weather features are sparse for non-target crops (sunflower-phase columns are NaN for wheat rows). For per-crop models this is fine.\n")
    md.append("- For Sprint 2: drop crops with n<20 from per-crop training; keep them only in 'all crops' aggregate.\n")

    out = REPORTS / "clean_dataset_audit.md"
    out.write_text("".join(md), encoding="utf-8")
    return out


def main():
    print("=" * 80)
    print("EDA AUDIT — clean dataset v2")
    print("=" * 80)
    df = load_dataset()
    print(f"Loaded {len(df)} rows × {len(df.columns)} cols")

    paths = {}
    paths["Target boxplot per crop"] = fig_target_boxplot(df)
    paths["Histograms (top-4 crops)"] = fig_target_hist_top4(df)
    paths["Year trend (median yield)"] = fig_target_year_trend(df)
    paths["NaN ratio per feature"] = fig_nan_ratio(df)
    paths["Feature × target correlation (per crop)"] = fig_corr_per_crop(df)

    md_path = write_md(df, paths)
    print("\nGenerated:")
    for k, p in paths.items():
        print(f"  - {p}")
    print(f"  - {md_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
