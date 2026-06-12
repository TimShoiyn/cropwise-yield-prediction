"""Generate final figures for the paper/slides from the result CSVs.

Outputs PNGs to reports/figures/:
  fig1_model_vs_cropwise_mape.png   model vs Cropwise as-of (MAPE) by crop, 1 Aug
  fig2_model_vs_cropwise_r2.png     same for R^2
  fig3_sentinel2_ablation.png       v26_base vs v27_s2 (R^2), all_crops & wheat
  fig4_management_ablation.png      v27/v28/v29 (R^2) by crop, 1 Aug
  fig5_data_scale.png               30 -> 528 fields with ground truth
  fig6_sunflower_scatter.png        predicted vs actual (our best crop)
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
COMP = Path("models_v2/asof_comparison")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

OURS = "#2a7ae2"
CROP = "#e0731a"
GREY = "#9aa0a6"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def fig_model_vs_cropwise() -> None:
    df = pd.read_csv(COMP / "asof_results_v30_policy.csv")
    order = ["sunflower", "wheat_combined", "all_crops", "barley"]
    labels = {"sunflower": "Подсолнечник", "wheat_combined": "Пшеница", "all_crops": "Все культуры", "barley": "Ячмень"}
    ours = df[df.model == "v30_policy"].set_index("scenario")
    cw = df[df.model == "cropwise_asof"].set_index("scenario")

    for metric, fname, title, lower_better in [
        ("mape", "fig1_model_vs_cropwise_mape.png", "MAPE, % (ниже — лучше)", True),
        ("r2", "fig2_model_vs_cropwise_r2.png", "R² (выше — лучше)", False),
    ]:
        x = np.arange(len(order))
        w = 0.38
        o = [ours.loc[s, metric] for s in order]
        c = [cw.loc[s, metric] for s in order]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(x - w / 2, o, w, label="Наша модель (v30)", color=OURS)
        ax.bar(x + w / 2, c, w, label="Cropwise as-of", color=CROP)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[s] for s in order])
        ax.set_ylabel(title)
        ax.set_title(f"Наша модель vs Cropwise на 1 августа — {title}")
        ax.legend()
        for i, (a, b) in enumerate(zip(o, c)):
            ax.text(i - w / 2, a, f"{a:.2f}" if not lower_better else f"{a:.0f}", ha="center", va="bottom", fontsize=9)
            ax.text(i + w / 2, b, f"{b:.2f}" if not lower_better else f"{b:.0f}", ha="center", va="bottom", fontsize=9)
        if metric == "r2":
            ax.axhline(0, color="black", lw=0.8)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)
        print("wrote", FIG / fname)


def fig_sentinel2() -> None:
    df = pd.read_csv(COMP / "asof_results_v27_s2.csv")
    sub = df[df.scenario.isin(["all_crops", "wheat_combined"]) & df.asof_tag.isin(["07_01", "08_01"])]
    cats, base, s2 = [], [], []
    for sc in ["all_crops", "wheat_combined"]:
        for tag, lab in [("07_01", "1 июл"), ("08_01", "1 авг")]:
            b = sub[(sub.scenario == sc) & (sub.asof_tag == tag) & (sub.model == "v26_base")]
            s = sub[(sub.scenario == sc) & (sub.asof_tag == tag) & (sub.model == "v27_s2")]
            if b.empty or s.empty:
                continue
            cats.append(f"{'Все' if sc=='all_crops' else 'Пшеница'}\n{lab}")
            base.append(float(b.r2.iloc[0]))
            s2.append(float(s.r2.iloc[0]))
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w / 2, base, w, label="Без Sentinel-2 (v26)", color=GREY)
    ax.bar(x + w / 2, s2, w, label="С Sentinel-2 (v27)", color=OURS)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("R²")
    ax.set_title("Вклад Sentinel-2 (NDRE/EVI/GCVI) в R²")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    for i, (a, b) in enumerate(zip(base, s2)):
        ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sentinel2_ablation.png")
    plt.close(fig)
    print("wrote", FIG / "fig3_sentinel2_ablation.png")


def fig_management() -> None:
    df = pd.read_csv(COMP / "asof_results_v29_pruned_management.csv")
    df = df[df.asof_tag == "08_01"]
    order = ["all_crops", "wheat_combined", "sunflower", "barley"]
    labels = {"all_crops": "Все", "wheat_combined": "Пшеница", "sunflower": "Подсолн.", "barley": "Ячмень"}
    models = [("v27_s2", "S2 без менедж.", GREY), ("v28_all_mgmt", "+ весь менедж.", OURS), ("v29_pruned_mgmt", "+ прунинг", CROP)]
    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for j, (m, lab, col) in enumerate(models):
        vals = []
        for s in order:
            r = df[(df.scenario == s) & (df.model == m)]
            vals.append(float(r.r2.iloc[0]) if not r.empty else np.nan)
        ax.bar(x + (j - 1) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in order])
    ax.set_ylabel("R²")
    ax.set_title("Менеджмент-фичи: эффект по культурам (1 августа, R²)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig4_management_ablation.png")
    plt.close(fig)
    print("wrote", FIG / "fig4_management_ablation.png")


def fig_data_scale() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(["Было\n(старый доступ)", "Стало\n(новый API)"], [30, 528], color=[GREY, OURS])
    ax.set_ylabel("Поля с фактическим урожаем")
    ax.set_title("Рост ground-truth: 30 → 528 полей (2021–2025)")
    for b, v in zip(bars, [30, 528]):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_data_scale.png")
    plt.close(fig)
    print("wrote", FIG / "fig5_data_scale.png")


def fig_sunflower_scatter() -> None:
    preds = pd.read_csv(COMP / "asof_predictions_v30_policy.csv")
    s = preds[(preds.scenario == "sunflower") & (preds.asof_tag == "08_01")].dropna(subset=["pred_v30_policy"])
    if s.empty:
        print("no sunflower preds; skip scatter")
        return
    y, p, cw = s.target_yield_t_ha, s.pred_v30_policy, s.cropwise_asof_t_ha
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(y, p, s=18, alpha=0.6, color=OURS, label="Наша модель")
    ax.scatter(y, cw, s=18, alpha=0.5, color=CROP, marker="x", label="Cropwise as-of")
    lim = [0, max(float(y.max()), float(p.max()), float(cw.max())) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="идеал")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Факт. урожай, т/га")
    ax.set_ylabel("Прогноз, т/га")
    ax.set_title("Подсолнечник, 1 августа: прогноз vs факт")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig6_sunflower_scatter.png")
    plt.close(fig)
    print("wrote", FIG / "fig6_sunflower_scatter.png")


def main() -> None:
    fig_model_vs_cropwise()
    fig_sentinel2()
    fig_management()
    fig_data_scale()
    fig_sunflower_scatter()
    print("\nAll figures in", FIG)


if __name__ == "__main__":
    main()
