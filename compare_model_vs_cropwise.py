"""
Compare model OOF predictions vs Cropwise predictions.

Inputs:
  - data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv
  - data_raw/fields.csv (id=field_id, name=field name)
  - models/oof_full_ext_cropwindow_allcrops_all_no_leak_reg.csv (OOF predictions)
  - data_raw/productivity_data.csv (Поле, Год, урожайность факт/прогноз ц/га)

Outputs:
  - models/models_vs_cropwise_joined_field_year.csv   (joined table, field_id x year)
  - models/models_vs_cropwise_metrics.csv             (long format metrics)

Notes on units:
  productivity_data.csv columns are labeled "ц/га" (centners/ha).
  This script detects whether values need dividing by 10 to match ML target scale (t/ha),
  based on the median ratio on matched rows.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
MODELS_DIR = Path("models")

ML_DATASET = DATA_PROCESSED / "ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv"
FIELDS_CSV = DATA_RAW / "fields.csv"
OOF_MODEL = MODELS_DIR / "oof_full_ext_cropwindow_allcrops_all_no_leak_reg.csv"
CROPWISE_CSV = DATA_RAW / "productivity_data.csv"

OUT_JOINED = MODELS_DIR / "models_vs_cropwise_joined_field_year.csv"
OUT_METRICS = MODELS_DIR / "models_vs_cropwise_metrics.csv"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _normalize_name(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("ё", "е")
    # collapse whitespace
    s = " ".join(s.split())
    return s


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


@dataclass
class UnitDecision:
    divide_by_10: bool
    ratio_median: float
    n_matched: int


def decide_units(ml_true: pd.Series, cropwise_true_raw: pd.Series) -> UnitDecision:
    """
    Decide whether cropwise values are in centners/ha (need /10) or already in t/ha.

    Logic:
      - compute median( cropwise_true_raw / ml_true ) on matched rows
      - if ratio is around 10 (8..12), we divide by 10
      - otherwise, keep as is
    """
    a = pd.to_numeric(ml_true, errors="coerce")
    b = pd.to_numeric(cropwise_true_raw, errors="coerce")
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n = int(mask.sum())
    if n < 20:
        # Not enough matched points: default to "no divide" but keep ratio for logging.
        ratio = float(np.nanmedian((b[mask] / a[mask]).to_numpy())) if n > 0 else float("nan")
        return UnitDecision(divide_by_10=False, ratio_median=ratio, n_matched=n)

    ratio = float(np.nanmedian((b[mask] / a[mask]).to_numpy()))
    divide = bool(8.0 <= ratio <= 12.0)
    return UnitDecision(divide_by_10=divide, ratio_median=ratio, n_matched=n)


def main() -> None:
    _require(ML_DATASET)
    _require(FIELDS_CSV)
    _require(OOF_MODEL)
    _require(CROPWISE_CSV)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load ML truth (field_id x year)
    # -----------------------------
    ml = pd.read_csv(ML_DATASET)
    need_ml = {"field_id", "year", "target_yield_t_ha"}
    if not need_ml.issubset(ml.columns):
        raise RuntimeError(f"ML dataset missing columns: {sorted(need_ml - set(ml.columns))}")
    ml = ml[["field_id", "year", "target_yield_t_ha"]].copy()
    ml["field_id"] = pd.to_numeric(ml["field_id"], errors="coerce").astype("Int64")
    ml["year"] = pd.to_numeric(ml["year"], errors="coerce").astype("Int64")
    ml["y_true"] = pd.to_numeric(ml["target_yield_t_ha"], errors="coerce")
    ml = ml.dropna(subset=["field_id", "year"]).copy()
    ml = ml.drop(columns=["target_yield_t_ha"])
    ml = ml.drop_duplicates(subset=["field_id", "year"], keep="last")

    # -----------------------------
    # Load OOF model predictions (field_id x year)
    # -----------------------------
    oof = pd.read_csv(OOF_MODEL)
    need_oof = {"field_id", "year", "model_pred_oof_t_ha"}
    if not need_oof.issubset(oof.columns):
        raise RuntimeError(f"OOF file missing columns: {sorted(need_oof - set(oof.columns))}")
    oof = oof[["field_id", "year", "model_pred_oof_t_ha"]].copy()
    oof["field_id"] = pd.to_numeric(oof["field_id"], errors="coerce").astype("Int64")
    oof["year"] = pd.to_numeric(oof["year"], errors="coerce").astype("Int64")
    oof["y_model_oof"] = pd.to_numeric(oof["model_pred_oof_t_ha"], errors="coerce")
    oof = oof.dropna(subset=["field_id", "year"]).copy()
    oof = oof.drop(columns=["model_pred_oof_t_ha"])
    oof = oof.drop_duplicates(subset=["field_id", "year"], keep="last")

    # -----------------------------
    # Load fields mapping (field_id -> name)
    # -----------------------------
    fields = pd.read_csv(FIELDS_CSV)
    if "id" not in fields.columns or "name" not in fields.columns:
        raise RuntimeError("fields.csv must contain columns: id, name")
    fields_map = fields[["id", "name"]].copy().rename(columns={"id": "field_id", "name": "field_name_fields"})
    fields_map["field_id"] = pd.to_numeric(fields_map["field_id"], errors="coerce").astype("Int64")
    fields_map["field_name_fields"] = fields_map["field_name_fields"].map(_normalize_name)
    fields_map = fields_map.dropna(subset=["field_id"]).copy()
    fields_map = fields_map.drop_duplicates(subset=["field_id"], keep="last")

    # -----------------------------
    # Load Cropwise CSV (field_name x year)
    # -----------------------------
    cw = pd.read_csv(CROPWISE_CSV)
    # required russian columns
    col_field = "Поле"
    col_year = "Год"
    col_fact = "урожайность факт ц/га"
    col_pred = "урожайность прогноз ц/га"
    for c in [col_field, col_year, col_fact, col_pred]:
        if c not in cw.columns:
            raise RuntimeError(f"productivity_data.csv missing column: {c}")

    cw = cw[[col_field, col_year, col_fact, col_pred]].copy()
    cw["field_name_cropwise"] = cw[col_field].map(_normalize_name)
    cw["year"] = pd.to_numeric(cw[col_year], errors="coerce").astype("Int64")
    cw["y_cropwise_fact_raw"] = pd.to_numeric(cw[col_fact], errors="coerce")
    cw["y_cropwise_pred_raw"] = pd.to_numeric(cw[col_pred], errors="coerce")
    cw = cw.drop(columns=[col_field, col_year, col_fact, col_pred])
    cw = cw.dropna(subset=["field_name_cropwise", "year"]).copy()

    # Map Cropwise field_name -> field_id using fields.csv name
    name_to_id = fields_map.dropna(subset=["field_name_fields"]).copy()
    name_to_id = name_to_id[name_to_id["field_name_fields"] != ""]
    name_to_id = name_to_id.drop_duplicates(subset=["field_name_fields"], keep="last")

    cw2 = cw.merge(
        name_to_id.rename(columns={"field_name_fields": "field_name_cropwise"}),
        on="field_name_cropwise",
        how="left",
        validate="m:1",
    )
    # Keep only rows that successfully mapped to field_id; unmatched names stay available in cw2,
    # but must be excluded from (field_id,year) merges.
    cw2 = cw2.dropna(subset=["field_id"]).copy()
    cw2["field_id"] = cw2["field_id"].astype("Int64")
    # Deduplicate in case the source has repeats for same field×year.
    cw2 = cw2.drop_duplicates(subset=["field_id", "year"], keep="last")

    # Decide units based on matched ML truth vs cropwise fact (where possible)
    probe = (
        ml.merge(cw2[["field_id", "year", "y_cropwise_fact_raw"]], on=["field_id", "year"], how="inner")
    )
    decision = decide_units(probe["y_true"], probe["y_cropwise_fact_raw"])

    if decision.divide_by_10:
        cw2["y_cropwise_pred"] = cw2["y_cropwise_pred_raw"] / 10.0
        cw2["y_cropwise_fact"] = cw2["y_cropwise_fact_raw"] / 10.0
        unit_note = "Detected ratio ~10x -> dividing Cropwise values by 10 (centners/ha -> t/ha)."
    else:
        cw2["y_cropwise_pred"] = cw2["y_cropwise_pred_raw"]
        cw2["y_cropwise_fact"] = cw2["y_cropwise_fact_raw"]
        unit_note = "Detected ratio not ~10x -> keeping Cropwise values as-is (assumed already in t/ha)."

    # -----------------------------
    # Build final joined table (field_id x year)
    # -----------------------------
    joined = ml.merge(oof, on=["field_id", "year"], how="left", validate="1:1")
    joined = joined.merge(fields_map[["field_id", "field_name_fields"]], on="field_id", how="left", validate="m:1")
    joined = joined.merge(
        cw2[["field_id", "year", "y_cropwise_pred", "field_name_cropwise"]],
        on=["field_id", "year"],
        how="left",
        validate="1:1",
    )

    # Save joined table
    joined.to_csv(OUT_JOINED, index=False)

    # -----------------------------
    # Metrics
    # -----------------------------
    def metrics_one(name: str, y_true_col: str, y_pred_col: str) -> dict:
        y_true = joined[y_true_col].to_numpy(dtype=float)
        y_pred = joined[y_pred_col].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_used = int(mask.sum())
        return {
            "comparison": name,
            "n_rows_total": int(len(joined)),
            "n_used": n_used,
            "r2": _safe_r2(y_true, y_pred),
            "rmse": _rmse(y_true, y_pred),
            "mape": _mape(y_true, y_pred),
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
        }

    metrics_rows = []
    metrics_rows.append(metrics_one("model_vs_actual", "y_true", "y_model_oof"))
    metrics_rows.append(metrics_one("cropwise_vs_actual", "y_true", "y_cropwise_pred"))
    metrics_rows.append(metrics_one("model_vs_cropwise", "y_cropwise_pred", "y_model_oof"))

    metrics_df = pd.DataFrame(metrics_rows)
    # add unit decision info to all rows
    metrics_df["cropwise_units_divide_by_10"] = decision.divide_by_10
    metrics_df["cropwise_fact_to_ml_median_ratio"] = decision.ratio_median
    metrics_df["cropwise_fact_to_ml_ratio_n"] = decision.n_matched
    metrics_df["notes"] = unit_note

    metrics_df.to_csv(OUT_METRICS, index=False)

    # Console summary
    print("=" * 80)
    print("MODEL vs CROPWISE SUMMARY")
    print("=" * 80)
    print(f"Joined table saved: {OUT_JOINED}")
    print(f"Metrics saved:      {OUT_METRICS}")
    print()
    print(f"Unit check on matched rows: n={decision.n_matched}, median_ratio(cropwise_fact_raw / ml_true)={decision.ratio_median:.3f}")
    print(unit_note)
    print()
    show = metrics_df[["comparison", "n_used", "r2", "rmse", "mape"]].copy()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

