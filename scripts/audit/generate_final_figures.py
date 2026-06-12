"""
Build all figures used in FINAL_REPORT_RU.md.
Outputs go to reports/figures/final/*.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures" / "final"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ---------------------------------------------------------------------------
# Figure 1: target distribution per crop in the v17 clean dataset
# ---------------------------------------------------------------------------
def fig_target_distribution() -> None:
    df = pd.read_csv(DATA_PROCESSED / "ml_dataset_clean_v17_v12_clean_asof_07_01.csv")
    crops_in_order = ["sunflower", "wheat_spring", "wheat_winter", "barley_spring", "oil_seed_raps_winter", "soya", "pea", "maize"]
    crops_present = [c for c in crops_in_order if c in df["standard_name"].unique()]
    data = [df[df["standard_name"] == c]["target_yield_t_ha"].values for c in crops_present]
    counts = [len(d) for d in data]
    labels = [f"{c}\n(n={n})" for c, n in zip(crops_present, counts)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    palette = ["#FFB000", "#785EF0", "#5A4FCF", "#1f77b4", "#2ca02c", "#d62728", "#8c564b", "#9467bd"]
    for patch, color in zip(bp["boxes"], palette[: len(crops_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)
    ax.set_ylabel("Урожайность, t/ha")
    ax.set_title("Распределение целевой переменной по культурам (v17 clean dataset, n=180)")
    ax.set_ylim(0, 7)
    fig.tight_layout()
    fig.savefig(FIGS / "01_target_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: NDVI seasonal curve example
# ---------------------------------------------------------------------------
def fig_ndvi_example() -> None:
    ndvi = pd.read_csv(DATA_RAW / "ndvi_timeseries.csv")
    ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
    ndvi = ndvi.dropna(subset=["date", "ndvi_mean"])
    field_id = 209
    sub = ndvi[ndvi["field_id"] == field_id].copy()
    sub["doy"] = sub["date"].dt.dayofyear
    sub["year"] = sub["date"].dt.year

    fig, ax = plt.subplots(figsize=(11, 5))
    years_show = [2020, 2021, 2022, 2023, 2024]
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(years_show)))
    for year, color in zip(years_show, palette):
        cur = sub[sub["year"] == year].sort_values("doy")
        if cur.empty:
            continue
        ax.plot(cur["doy"], cur["ndvi_mean"], label=f"{year}", color=color, alpha=0.8, linewidth=1.5)

    for d, label in [(182, "1 июл"), (213, "1 авг"), (244, "1 сен")]:
        ax.axvline(d, color="red", linestyle="--", alpha=0.5)
        ax.text(d + 1, 0.95, label, rotation=90, va="top", color="red", alpha=0.7, fontsize=9)

    ax.set_xlim(60, 320)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Day of year (DOY)")
    ax.set_ylabel("NDVI (mean over field polygon)")
    ax.set_title(f"Сезонная кривая NDVI по годам, поле id={field_id}")
    ax.legend(title="Год", loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "02_ndvi_seasonal.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: walk-forward CV schema
# ---------------------------------------------------------------------------
def fig_wf_schema() -> None:
    years = list(range(2018, 2026))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, test_year in enumerate(years[1:], start=1):
        train_years = [y for y in years if y < test_year]
        for y in train_years:
            ax.barh(i, 1, left=y, color="#1f77b4", alpha=0.7)
        ax.barh(i, 1, left=test_year, color="#d62728", alpha=0.85)
    ax.set_yticks(list(range(1, len(years))))
    ax.set_yticklabels([f"Fold {i}: test={years[i]}" for i in range(1, len(years))])
    ax.set_xticks(years)
    ax.set_xlabel("Год")
    ax.set_title("Walk-forward кросс-валидация (синий = train, красный = test)")
    fig.tight_layout()
    fig.savefig(FIGS / "03_walk_forward.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: ML / Cropwise / naive baseline MAPE comparison
# ---------------------------------------------------------------------------
def fig_metric_comparison() -> None:
    base = pd.read_csv(REPORTS / "microscope" / "baselines.csv")
    scenarios = ["sunflower", "wheat_combined", "all_crops"]
    asofs = ["1 Jul", "1 Aug", "1 Sep"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, metric in zip(axes, ["mape", "mae", "rmse", "r2"]):
        x = np.arange(len(asofs))
        width = 0.13
        bars = {
            "ML": (f"ml_{metric}", "#2ca02c"),
            "Cropwise": (f"cw_{metric}", "#d62728"),
            "global_mean": (f"global_mean_{metric}", "#7f7f7f"),
            "crop_mean": (f"crop_mean_{metric}", "#bcbd22"),
            "field_mean": (f"field_mean_{metric}", "#17becf"),
            "last_year": (f"last_year_{metric}", "#8c564b"),
        }
        for sc_idx, sc in enumerate(scenarios):
            sub = base[base["scenario"] == sc].set_index("asof_label").reindex(asofs)
            for j, (label, (col, color)) in enumerate(bars.items()):
                values = sub[col].values
                offset = (j - 2.5) * width
                ax.bar(x + offset + sc_idx * (width * 6.5), values, width, label=label if sc_idx == 0 else None, color=color, alpha=0.85)
        ax.set_xticks(x + width * 6.5 * 1)
        ax.set_xticklabels([s.split()[1] if " " in s else s for s in asofs])
        ax.set_title(metric.upper())
        ax.set_xlabel("Дата прогноза (по сценариям sunflower / wheat_combined / all_crops)")
        if metric == "mape":
            ax.set_ylabel("%")
            ax.set_ylim(0, max(80, base[[c for c in base.columns if c.endswith("_mape") and not c.startswith("cw")]].max().max() * 1.05))
        elif metric == "r2":
            ax.set_ylim(-2.5, 1.0)
            ax.axhline(0, color="black", linewidth=0.7)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("ML vs Cropwise vs naive baselines — все 4 метрики", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "04_metric_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: per-year MAPE for ML and Cropwise (1 Aug)
# ---------------------------------------------------------------------------
def fig_per_year() -> None:
    py_df = pd.read_csv(REPORTS / "microscope" / "per_year_errors.csv")
    sub = py_df[py_df["asof_tag"] == "08_01"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    scenarios = ["all_crops", "wheat_combined", "sunflower"]
    for ax, sc in zip(axes, scenarios):
        cur = sub[sub["scenario"] == sc].sort_values("year")
        ax.plot(cur["year"], cur["ml_mape"], "o-", label="ML", color="#2ca02c", markersize=8, linewidth=2)
        ax.plot(cur["year"], cur["cw_mape"], "s-", label="Cropwise", color="#d62728", markersize=8, linewidth=2)
        ax.set_title(f"{sc} (1 Aug)")
        ax.set_xlabel("Тестовый год")
        ax.legend()
        ax.set_ylim(0, min(120, cur[["ml_mape", "cw_mape"]].max().max() * 1.1))
    axes[0].set_ylabel("MAPE, %")
    fig.suptitle("Ошибка по годам (1 Aug as-of, walk-forward OOF)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "05_per_year.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: top-15 feature importance (after CW leakage fix)
# ---------------------------------------------------------------------------
def fig_feature_importance() -> None:
    fi = pd.read_csv(REPORTS / "microscope" / "feature_importance.csv")
    scenarios = ["all_crops", "wheat_combined", "sunflower"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, sc in zip(axes, scenarios):
        sub = fi[fi["scenario"] == sc].sort_values("importance", ascending=True).tail(15)
        ax.barh(sub["feature"], sub["importance"], color="#1f77b4", alpha=0.85)
        ax.set_title(f"Top-15 features for {sc}")
        ax.set_xlabel("CatBoost importance")
    fig.suptitle("Feature importance после фикса leakage (CatBoost in-sample, 1 Aug)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "06_feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: in-sample vs OOF gap
# ---------------------------------------------------------------------------
def fig_overfit() -> None:
    io = pd.read_csv(REPORTS / "microscope" / "insample_vs_oof.csv")
    sub = io[io["asof_tag"] == "08_01"]

    scenarios = sub["scenario"].tolist()
    in_r2 = sub["in_sample_r2"].values
    oof_r2 = sub["oof_r2"].values
    in_mape = sub["in_sample_mape"].values
    oof_mape = sub["oof_mape"].values

    x = np.arange(len(scenarios))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, in_r2, width, label="In-sample (train data)", color="#2ca02c")
    axes[0].bar(x + width / 2, oof_r2, width, label="OOF (walk-forward)", color="#d62728")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].set_title("R²: in-sample vs OOF")
    axes[0].legend()

    axes[1].bar(x - width / 2, in_mape, width, label="In-sample", color="#2ca02c")
    axes[1].bar(x + width / 2, oof_mape, width, label="OOF", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios)
    axes[1].set_title("MAPE %: in-sample vs OOF")
    axes[1].legend()

    fig.suptitle("Диагностика overfitting (1 Aug)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "07_overfit_gap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_target_distribution()
    print("[1] target distribution")
    fig_ndvi_example()
    print("[2] NDVI example")
    fig_wf_schema()
    print("[3] walk-forward")
    fig_metric_comparison()
    print("[4] metric comparison")
    fig_per_year()
    print("[5] per-year")
    fig_feature_importance()
    print("[6] feature importance")
    fig_overfit()
    print("[7] overfit")
    print(f"\nAll figures saved to {FIGS}")


if __name__ == "__main__":
    main()
