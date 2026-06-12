"""
Build Cropwise as-of predictions from productivity_estimate_histories.csv and evaluate vs ML truth.

Inputs:
  - data_raw/productivity_estimate_histories.csv
      columns: field_id, year, estimate_history (string with dict-like date->value)
  - data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv
      columns: field_id, year, target_yield_t_ha

Outputs:
  - models/cropwise_asof_joined.csv
  - models/cropwise_asof_metrics.csv
"""

from __future__ import annotations

import os
import ast
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
MODELS_DIR = Path("models")

HIST_CSV = DATA_RAW / "productivity_estimate_histories.csv"
ML_CSV = DATA_PROCESSED / "ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv"

OUT_JOINED = MODELS_DIR / "cropwise_asof_joined.csv"
OUT_METRICS = MODELS_DIR / "cropwise_asof_metrics.csv"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _parse_history(s: Any) -> dict[pd.Timestamp, float]:
    """
    estimate_history comes like:
      "{'2025-05-24': 26.635, '2025-05-31': 24.989, ...}"
    i.e. Python dict string (single quotes), not strict JSON.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    if isinstance(s, dict):
        raw = s
    else:
        txt = str(s).strip()
        if not txt:
            return {}
        raw = None
        # Try JSON first (just in case), then ast.literal_eval
        try:
            raw = json.loads(txt)
        except Exception:
            try:
                raw = ast.literal_eval(txt)
            except Exception:
                return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[pd.Timestamp, float] = {}
    for k, v in raw.items():
        try:
            dt = pd.to_datetime(k, errors="coerce")
        except Exception:
            dt = pd.NaT
        if pd.isna(dt):
            continue
        try:
            val = float(v)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        out[pd.Timestamp(dt).normalize()] = val
    return out


def _asof_value(history: dict[pd.Timestamp, float], cutoff: pd.Timestamp) -> float:
    if not history:
        return float("nan")
    cutoff = pd.Timestamp(cutoff).normalize()
    dates = sorted(history.keys())
    # last date <= cutoff
    idx = np.searchsorted(np.array(dates, dtype="datetime64[ns]"), np.datetime64(cutoff), side="right") - 1
    if idx < 0:
        return float("nan")
    return float(history[dates[int(idx)]])


def _final_value(history: dict[pd.Timestamp, float]) -> float:
    if not history:
        return float("nan")
    dt = max(history.keys())
    return float(history[dt])


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


def main() -> None:
    _require(HIST_CSV)
    _require(ML_CSV)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ML truth
    ml = pd.read_csv(ML_CSV)
    need_ml = {"field_id", "year", "target_yield_t_ha"}
    if not need_ml.issubset(ml.columns):
        raise RuntimeError(f"ML dataset missing columns: {sorted(need_ml - set(ml.columns))}")
    ml = ml[["field_id", "year", "target_yield_t_ha"]].copy()
    ml["field_id"] = pd.to_numeric(ml["field_id"], errors="coerce").astype("Int64")
    ml["year"] = pd.to_numeric(ml["year"], errors="coerce").astype("Int64")
    ml["y_true"] = pd.to_numeric(ml["target_yield_t_ha"], errors="coerce")
    ml = ml.drop(columns=["target_yield_t_ha"])
    ml = ml.dropna(subset=["field_id", "year"]).drop_duplicates(subset=["field_id", "year"], keep="last")

    # Histories
    hist = pd.read_csv(HIST_CSV)
    need_h = {"field_id", "year", "estimate_history"}
    if not need_h.issubset(hist.columns):
        raise RuntimeError(f"Histories CSV missing columns: {sorted(need_h - set(hist.columns))}")
    hist = hist[["field_id", "year", "estimate_history"]].copy()
    hist["field_id"] = pd.to_numeric(hist["field_id"], errors="coerce").astype("Int64")
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce").astype("Int64")
    hist = hist.dropna(subset=["field_id", "year"]).copy()

    # Deduplicate: if multiple rows per (field_id,year), keep last
    hist = hist.drop_duplicates(subset=["field_id", "year"], keep="last")

    # Build as-of columns
    rows = []
    for _, r in hist.iterrows():
        field_id = int(r["field_id"])
        year = int(r["year"])
        h = _parse_history(r["estimate_history"])
        cutoff_0701 = pd.Timestamp(year=year, month=7, day=1)
        cutoff_0801 = pd.Timestamp(year=year, month=8, day=1)
        cutoff_0901 = pd.Timestamp(year=year, month=9, day=1)
        rows.append(
            {
                "field_id": field_id,
                "year": year,
                "cropwise_pred_asof_0701": _asof_value(h, cutoff_0701),
                "cropwise_pred_asof_0801": _asof_value(h, cutoff_0801),
                "cropwise_pred_asof_0901": _asof_value(h, cutoff_0901),
                "cropwise_pred_final": _final_value(h),
            }
        )

    asof_df = pd.DataFrame(rows)

    # Join to ML truth
    joined = ml.merge(asof_df, on=["field_id", "year"], how="left", validate="1:1")
    joined.to_csv(OUT_JOINED, index=False)

    # Metrics per asof tag
    metric_rows = []
    for tag, col in [
        ("0701", "cropwise_pred_asof_0701"),
        ("0801", "cropwise_pred_asof_0801"),
        ("0901", "cropwise_pred_asof_0901"),
        ("final", "cropwise_pred_final"),
    ]:
        y_true = joined["y_true"].to_numpy(dtype=float)
        y_pred = joined[col].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        metric_rows.append(
            {
                "asof_tag": tag,
                "n_used": int(mask.sum()),
                "r2": _safe_r2(y_true, y_pred),
                "rmse": _rmse(y_true, y_pred),
                "mape": _mape(y_true, y_pred),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_METRICS, index=False)

    # Console summary
    print("=" * 80)
    print("CROPWISE AS-OF EVALUATION")
    print("=" * 80)
    print(f"Joined saved:  {OUT_JOINED}")
    print(f"Metrics saved: {OUT_METRICS}")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

