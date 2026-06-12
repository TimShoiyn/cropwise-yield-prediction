"""
Sprint 5.3+ — Simple ensemble of v4 (anomaly) and v5 (GDD-aligned).

Average the OOF predictions from per_row_predictions_v4.csv and per_row_predictions_v5.csv,
re-compute metrics. Compare to v4, v5 and Cropwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"


def metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    if len(y) < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "mape": float(np.mean(np.abs((y - p) / np.where(y > 0, y, np.nan))) * 100),
    }


def main():
    p4 = pd.read_csv(COMP_DIR / "per_row_predictions_v4.csv")
    p5 = pd.read_csv(COMP_DIR / "per_row_predictions_v5.csv")
    keys = ["scenario", "asof_tag", "field_id", "year", "crop_id"]
    merged = p4.merge(
        p5[keys + ["ml_v5_t_ha"]],
        on=keys, how="inner",
    )
    merged["ml_ens_t_ha"] = (merged["ml_v4_t_ha"] + merged["ml_v5_t_ha"]) / 2.0

    rows = []
    for (sc, tag), g in merged.groupby(["scenario", "asof_tag"]):
        # ml-only metrics
        m_v4 = metrics(g["target_yield_t_ha"], g["ml_v4_t_ha"])
        m_v5 = metrics(g["target_yield_t_ha"], g["ml_v5_t_ha"])
        m_e = metrics(g["target_yield_t_ha"], g["ml_ens_t_ha"])
        # cropwise metrics on rows where it exists
        g_cw = g.dropna(subset=["cropwise_asof_t_ha"])
        m_cw = metrics(g_cw["target_yield_t_ha"], g_cw["cropwise_asof_t_ha"]) if len(g_cw) else {}
        rows.append({
            "scenario": sc, "asof_tag": tag, "n": len(g), "n_cw": len(g_cw),
            "v4_mape": m_v4["mape"], "v5_mape": m_v5["mape"], "ens_mape": m_e["mape"],
            "v4_r2": m_v4["r2"],   "v5_r2": m_v5["r2"],   "ens_r2": m_e["r2"],
            "v4_rmse": m_v4["rmse"], "v5_rmse": m_v5["rmse"], "ens_rmse": m_e["rmse"],
            "cw_mape": m_cw.get("mape", float("nan")),
            "cw_r2": m_cw.get("r2", float("nan")),
            "cw_rmse": m_cw.get("rmse", float("nan")),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(COMP_DIR / "asof_results_ens.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # plot
    plt.figure(figsize=(10, 6))
    for sc in summary["scenario"].unique():
        s = summary[summary["scenario"] == sc].sort_values("asof_tag")
        plt.plot(s["asof_tag"], s["v4_mape"], marker="o", lw=1.6, label=f"v4 — {sc}")
        plt.plot(s["asof_tag"], s["v5_mape"], marker="^", lw=1.6, label=f"v5 — {sc}")
        plt.plot(s["asof_tag"], s["ens_mape"], marker="*", lw=2.5, label=f"ensemble — {sc}")
        plt.plot(s["asof_tag"], s["cw_mape"], marker="s", lw=1.4, ls="--", color="grey", label=f"Cropwise — {sc}")
    plt.title("Ensemble v4+v5 vs single models vs Cropwise")
    plt.xlabel("As-of"); plt.ylabel("MAPE, %"); plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p = FIGS / "asof_mape_ensemble.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"\nSaved {p}")
    print(f"Saved {COMP_DIR / 'asof_results_ens.csv'}")


if __name__ == "__main__":
    main()
