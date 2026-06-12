"""
Generate final v22 figures for dissertation/professor discussion.

Outputs:
  reports/figures/v22_final/*.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

FIGS = Path("reports/figures/v22_final")
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 10


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_01_r2_decomposition() -> None:
    df = pd.read_csv("reports/microscope/r2_decomposition.csv")
    sub = df[(df["asof"] == "1 Aug") & (df["model"].isin(["ml_peer", "cropwise"]))]
    scenarios = ["all_crops", "wheat_combined", "sunflower"]
    labels = {"ml_peer": "ML peer/v21", "cropwise": "Cropwise"}
    colors = {"pooled_r2": "#1f77b4", "within_crop_year_r2": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, sc in zip(axes, scenarios):
        cur = sub[sub["scenario"] == sc]
        x = np.arange(len(cur))
        ax.bar(x - 0.18, cur["pooled_r2"], width=0.36, color=colors["pooled_r2"], label="Pooled R²")
        ax.bar(x + 0.18, cur["within_crop_year_r2"], width=0.36, color=colors["within_crop_year_r2"], label="Within-crop-year R²")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in cur["model"]], rotation=20, ha="right")
        ax.set_title(sc)
        ax.set_ylim(-0.6, 0.8)
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("R²")
    fig.suptitle("Обычный R² vs честное ранжирование полей (1 Aug)", y=1.05)
    save(fig, "01_pooled_vs_within_crop_year_r2.png")


def fig_02_v22_wheat_mape() -> None:
    df = pd.read_csv("models_v2/asof_comparison/asof_results_v22_core_history.csv")
    sub = df[(df["scenario"] == "wheat_combined") & (df["model"].isin(["v22_core", "cropwise"]))]
    asofs = ["1 Jul", "1 Aug", "1 Sep"]
    x = np.arange(len(asofs))
    width = 0.36
    core = sub[sub["model"] == "v22_core"].set_index("asof").reindex(asofs)
    cw = sub[sub["model"] == "cropwise"].set_index("asof").reindex(asofs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, core["mape"], width, label="v22_core", color="#2ca02c")
    ax.bar(x + width / 2, cw["mape"], width, label="Cropwise", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(asofs)
    ax.set_ylabel("MAPE, %")
    ax.set_title("Пшеница: v22_core vs Cropwise по MAPE")
    ax.legend()
    for i, v in enumerate(core["mape"]):
        ax.text(i - width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(cw["mape"]):
        ax.text(i + width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    save(fig, "02_v22_wheat_mape_vs_cropwise.png")


def fig_03_noise_ceiling() -> None:
    df = pd.read_csv("reports/microscope/audit3b_noise_ceiling.csv")
    df = df[df["crop"].isin(["wheat_winter", "wheat_spring", "sunflower", "barley_spring"])]
    colors = ["#5A4FCF" if "wheat" in c else "#FFB000" if c == "sunflower" else "#7f7f7f" for c in df["crop"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["crop"], df["max_r2"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Estimated max achievable pooled R²")
    ax.set_title("Потолок шума при текущих признаках")
    ax.set_ylim(-0.2, 1.0)
    for i, v in enumerate(df["max_r2"]):
        ax.text(i, v + 0.035, f"{v:.2f}", ha="center", fontsize=10)
    save(fig, "03_noise_ceiling_by_crop.png")


def fig_04_pruning_effect() -> None:
    df = pd.read_csv("reports/microscope/audit3b_pruning.csv")
    scenarios = ["all", "wheat", "sunflower"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [(axes[0], "r2", "R²"), (axes[1], "mape", "MAPE, %")]:
        x = np.arange(len(scenarios))
        full = df[df["set"] == "full"].set_index("scenario").reindex(scenarios)
        core = df[df["set"] == "core"].set_index("scenario").reindex(scenarios)
        ax.bar(x - 0.18, full[metric], width=0.36, label="full 149 features", color="#7f7f7f")
        ax.bar(x + 0.18, core[metric], width=0.36, label="core 24 features", color="#2ca02c")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Прунинг признаков: compact core лучше sparse full", y=1.04)
    save(fig, "04_feature_pruning_effect.png")


def fig_05_lofo() -> None:
    df = pd.read_csv("reports/microscope/audit3b_lofo.csv")
    scenarios = df["scenario"].tolist()
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, df["lofo_r2"], width=0.36, label="LOFO pooled R²", color="#1f77b4")
    ax.bar(x + 0.18, df["lofo_wcy_r2"], width=0.36, label="LOFO within-crop-year R²", color="#d62728")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("R²")
    ax.set_title("Leave-one-field-out: перенос на новое поле vs ранжирование внутри года")
    ax.legend()
    save(fig, "05_lofo_spatial_generalization.png")


def fig_06_ndvi_domain_shift() -> None:
    peers = pd.read_csv("data_processed/ml_dataset_v18_peers_strict_asof_08_01.csv")
    ours = pd.read_csv("data_processed/ml_dataset_clean_v17_v12_clean_asof_08_01.csv")
    p = peers["ndvi_mean_asof"].dropna()
    o = ours["ndvi_mean_asof"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(p, bins=40, alpha=0.55, density=True, label=f"peers mean={p.mean():.3f}", color="#1f77b4")
    ax.hist(o, bins=25, alpha=0.55, density=True, label=f"our fields mean={o.mean():.3f}", color="#d62728")
    ax.set_xlabel("NDVI mean as-of 1 Aug")
    ax.set_ylabel("Density")
    ax.set_title("Domain shift: наши поля систематически зеленее peer-полей")
    ax.legend()
    save(fig, "06_ndvi_domain_shift.png")


def main() -> None:
    fig_01_r2_decomposition()
    fig_02_v22_wheat_mape()
    fig_03_noise_ceiling()
    fig_04_pruning_effect()
    fig_05_lofo()
    fig_06_ndvi_domain_shift()
    print(f"Wrote figures to {FIGS}")
    for p in sorted(FIGS.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
