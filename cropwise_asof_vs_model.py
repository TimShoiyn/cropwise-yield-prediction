"""
Compare Cropwise as-of predictions vs our model OOF predictions.

Inputs:
  - models/cropwise_asof_joined.csv
      columns: field_id, year, y_true, cropwise_pred_asof_0701/0801/0901(/final)
  - models/oof_full_ext_cropwindow_allcrops_all_no_leak_reg.csv
      columns: field_id, year, model_pred_oof_t_ha, target_yield_t_ha

Outputs:
  - models/cropwise_asof_vs_model_joined.csv
  - models/cropwise_asof_vs_model_metrics.csv   (rows: asof_tag × {model,cropwise})
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

MODELS_DIR = Path("models")

ASOF_JOINED = MODELS_DIR / "cropwise_asof_joined.csv"
OOF_MODEL = MODELS_DIR / "oof_full_ext_cropwindow_allcrops_all_no_leak_reg.csv"

OUT_JOINED = MODELS_DIR / "cropwise_asof_vs_model_joined.csv"
OUT_METRICS = MODELS_DIR / "cropwise_asof_vs_model_metrics.csv"


def _require(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray, *, min_n: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < min_n:
        return float("nan")
    return float(r2_score(y_true[mask], y_pred[mask]))


def _metrics_one(df: pd.DataFrame, *, asof_tag: str, source: str, y_pred_col: str) -> dict:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "asof_tag": asof_tag,
        "source": source,  # model or cropwise
        "n_used": int(mask.sum()),
        "r2": _safe_r2(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "y_pred_col": y_pred_col,
    }


def main() -> None:
    _require(ASOF_JOINED)
    _require(OOF_MODEL)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    asof = pd.read_csv(ASOF_JOINED)
    need_a = {"field_id", "year", "y_true", "cropwise_pred_asof_0701", "cropwise_pred_asof_0801", "cropwise_pred_asof_0901"}
    if not need_a.issubset(asof.columns):
        raise RuntimeError(f"cropwise_asof_joined.csv missing columns: {sorted(need_a - set(asof.columns))}")
    asof = asof[["field_id", "year", "y_true", "cropwise_pred_asof_0701", "cropwise_pred_asof_0801", "cropwise_pred_asof_0901"]].copy()
    asof["field_id"] = pd.to_numeric(asof["field_id"], errors="coerce").astype("Int64")
    asof["year"] = pd.to_numeric(asof["year"], errors="coerce").astype("Int64")
    asof["y_true"] = pd.to_numeric(asof["y_true"], errors="coerce")

    oof = pd.read_csv(OOF_MODEL)
    need_o = {"field_id", "year", "model_pred_oof_t_ha"}
    if not need_o.issubset(oof.columns):
        raise RuntimeError(f"OOF file missing columns: {sorted(need_o - set(oof.columns))}")
    oof = oof[["field_id", "year", "model_pred_oof_t_ha"]].copy()
    oof["field_id"] = pd.to_numeric(oof["field_id"], errors="coerce").astype("Int64")
    oof["year"] = pd.to_numeric(oof["year"], errors="coerce").astype("Int64")
    oof["y_model_oof"] = pd.to_numeric(oof["model_pred_oof_t_ha"], errors="coerce")
    oof = oof.drop(columns=["model_pred_oof_t_ha"])

    joined = asof.merge(oof, on=["field_id", "year"], how="inner", validate="1:1")
    joined.to_csv(OUT_JOINED, index=False)

    metrics_rows: list[dict] = []
    for tag, col in [("0701", "cropwise_pred_asof_0701"), ("0801", "cropwise_pred_asof_0801"), ("0901", "cropwise_pred_asof_0901")]:
        # model vs actual (same model column for all tags, but we repeat row per tag for the requested layout)
        metrics_rows.append(_metrics_one(joined, asof_tag=tag, source="model", y_pred_col="y_model_oof"))
        metrics_rows.append(_metrics_one(joined, asof_tag=tag, source="cropwise", y_pred_col=col))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT_METRICS, index=False)

    # Console summary: who is better by RMSE (lower is better) and by R2 (higher is better)
    print("=" * 80)
    print("CROPWISE AS-OF vs MODEL (OOF) SUMMARY")
    print("=" * 80)
    print(f"Joined saved:  {OUT_JOINED}")
    print(f"Metrics saved: {OUT_METRICS}")
    print()
    print(metrics[["asof_tag", "source", "n_used", "r2", "rmse", "mape"]].to_string(index=False))
    print()

    for tag in ["0701", "0801", "0901"]:
        m = metrics[(metrics["asof_tag"] == tag) & (metrics["source"] == "model")].iloc[0]
        c = metrics[(metrics["asof_tag"] == tag) & (metrics["source"] == "cropwise")].iloc[0]
        # Decide winner by RMSE primarily
        if np.isfinite(m["rmse"]) and np.isfinite(c["rmse"]):
            if m["rmse"] < c["rmse"]:
                winner = "model"
                delta = float(c["rmse"] - m["rmse"])
            else:
                winner = "cropwise"
                delta = float(m["rmse"] - c["rmse"])
            print(f"As-of {tag}: winner by RMSE = {winner} (RMSE gap = {delta:.3f} t/ha)")
        else:
            print(f"As-of {tag}: cannot compare RMSE (missing values).")


if __name__ == "__main__":
    main()

