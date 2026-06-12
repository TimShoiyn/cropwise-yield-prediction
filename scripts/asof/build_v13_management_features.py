"""
v13 = v12 corrected target + management/as-of operation features.

Adds features from operations.csv available before forecast date, excluding
harvesting operations to avoid harvest-time leakage.

Outputs:
  data_processed/ml_dataset_clean_v13_asof_{tag}.csv
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]
NUTRIENTS = ["N", "P2O5", "K2O", "S", "Mg", "B", "Zn"]
CUSTOM_FLAGS = [
    "x_custom_gerbicidnaya_khimobrabotka",
    "x_custom_insekticidnaya_khimobrabotka",
    "x_custom_podkormka",
    "x_custom_fungicidnaya_khimobrabotka",
]


def parse_mix(value: Any) -> list[dict[str, Any]]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def load_operations() -> pd.DataFrame:
    cols = [
        "id",
        "field_id",
        "season",
        "operation_type",
        "operation_subtype",
        "applications_type",
        "planned_area",
        "completed_area",
        "covered_area",
        "actual_start_datetime",
        "completed_date",
        "completed_datetime",
        "planned_start_date",
        "planned_end_date",
        "application_mix_items",
    ] + CUSTOM_FLAGS
    ops = pd.read_csv(DATA_RAW / "operations.csv", usecols=cols, low_memory=False)
    ops["field_id"] = pd.to_numeric(ops["field_id"], errors="coerce").astype("Int64")
    ops["season"] = pd.to_numeric(ops["season"], errors="coerce").astype("Int64")
    ops = ops.dropna(subset=["field_id", "season"]).copy()
    ops["field_id"] = ops["field_id"].astype(int)
    ops["season"] = ops["season"].astype(int)

    for col in ["planned_area", "completed_area", "covered_area"]:
        ops[col] = pd.to_numeric(ops[col], errors="coerce")
    for col in ["actual_start_datetime", "completed_datetime"]:
        ops[col] = pd.to_datetime(ops[col], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["completed_date", "planned_start_date", "planned_end_date"]:
        ops[col] = pd.to_datetime(ops[col], errors="coerce")
    ops["op_date"] = ops["actual_start_datetime"].fillna(ops["completed_datetime"]).fillna(ops["completed_date"]).fillna(ops["planned_start_date"])
    for col in CUSTOM_FLAGS:
        ops[col] = ops[col].astype(str).str.lower().eq("true")
    return ops


def load_fertilizers() -> pd.DataFrame:
    fert = pd.read_csv(DATA_RAW / "fertilizers_npk.csv")
    fert = fert.rename(columns={"id": "applicable_id"})
    for col in NUTRIENTS:
        fert[col] = pd.to_numeric(fert[col], errors="coerce").fillna(0.0)
    return fert[["applicable_id", "name", *NUTRIENTS]]


def summarize_management(row: pd.Series, ops: pd.DataFrame, fert: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float | int]:
    fid = int(row["field_id"])
    year = int(row["year"])
    g_all = ops[(ops["field_id"] == fid) & (ops["season"] == year)].copy()
    g = g_all[
        g_all["op_date"].notna()
        & (g_all["op_date"] <= as_of)
        & ~g_all["operation_type"].astype(str).eq("harvesting")
    ].sort_values("op_date")

    out: dict[str, float | int] = {
        "mgmt_ops_n_asof": int(len(g)),
        "mgmt_has_ops_asof": int(len(g) > 0),
    }
    if g.empty:
        return out

    out["mgmt_first_op_doy"] = int(g["op_date"].min().dayofyear)
    out["mgmt_last_op_doy"] = int(g["op_date"].max().dayofyear)
    out["mgmt_days_since_last_op"] = int((as_of - g["op_date"].max()).days)
    out["mgmt_completed_area_sum"] = float(g["completed_area"].fillna(0).sum())
    out["mgmt_completed_area_mean"] = float(g["completed_area"].mean()) if g["completed_area"].notna().any() else np.nan
    out["mgmt_completed_area_max"] = float(g["completed_area"].max()) if g["completed_area"].notna().any() else np.nan
    for operation_type in ["application", "soil", "other"]:
        out[f"mgmt_op_{operation_type}_n"] = int(g["operation_type"].astype(str).eq(operation_type).sum())
    for subtype in ["harrowing", "discing", "other"]:
        out[f"mgmt_subtype_{subtype}_n"] = int(g["operation_subtype"].astype(str).eq(subtype).sum())
    for col in CUSTOM_FLAGS:
        short = col.replace("x_custom_", "mgmt_flag_").replace("khimobrabotka", "chem").replace("gerbicidnaya", "herbicide").replace("insekticidnaya", "insecticide").replace("fungicidnaya", "fungicide").replace("podkormka", "topdress")
        out[f"{short}_n"] = int(g[col].sum())
        out[f"{short}_any"] = int(g[col].any())

    mix_rows: list[dict[str, Any]] = []
    for _, op in g.iterrows():
        for item in parse_mix(op.get("application_mix_items")):
            item = dict(item)
            item["op_date"] = op["op_date"]
            item["operation_type"] = op["operation_type"]
            mix_rows.append(item)
    if not mix_rows:
        return out

    mix = pd.DataFrame(mix_rows)
    mix["fact_rate"] = pd.to_numeric(mix.get("fact_rate"), errors="coerce").fillna(0.0)
    mix["planned_rate"] = pd.to_numeric(mix.get("planned_rate"), errors="coerce").fillna(0.0)
    mix["fact_amount"] = pd.to_numeric(mix.get("fact_amount"), errors="coerce").fillna(0.0)
    mix["applicable_id"] = pd.to_numeric(mix.get("applicable_id"), errors="coerce").astype("Int64")
    mix["applicable_type"] = mix.get("applicable_type", pd.Series(index=mix.index, dtype=str)).fillna("unknown").astype(str)

    for applicable_type in ["Fertilizer", "Seed", "Chemical"]:
        sub = mix[mix["applicable_type"] == applicable_type]
        key = applicable_type.lower()
        out[f"mgmt_{key}_items_n"] = int(len(sub))
        out[f"mgmt_{key}_fact_rate_sum"] = float(sub["fact_rate"].sum())
        out[f"mgmt_{key}_fact_rate_mean"] = float(sub["fact_rate"].mean()) if len(sub) else 0.0
        out[f"mgmt_{key}_fact_rate_max"] = float(sub["fact_rate"].max()) if len(sub) else 0.0
        out[f"mgmt_{key}_fact_amount_sum"] = float(sub["fact_amount"].sum())
        if len(sub):
            out[f"mgmt_{key}_last_doy"] = int(sub["op_date"].max().dayofyear)
            out[f"mgmt_{key}_days_since_last"] = int((as_of - sub["op_date"].max()).days)

    fert_mix = mix[mix["applicable_type"] == "Fertilizer"].merge(fert, on="applicable_id", how="left")
    if not fert_mix.empty:
        for nutrient in NUTRIENTS:
            pct = pd.to_numeric(fert_mix[nutrient], errors="coerce").fillna(0.0)
            # Fact rate is usually product rate per ha. Nutrient kg/ha proxy = rate * nutrient% / 100.
            out[f"mgmt_fert_{nutrient}_rate_sum"] = float((fert_mix["fact_rate"] * pct / 100.0).sum())
            out[f"mgmt_fert_{nutrient}_rate_max"] = float((fert_mix["fact_rate"] * pct / 100.0).max())
    return out


def build_one(tag: str, ops: pd.DataFrame, fert: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v12_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v13_asof_{tag}.csv"
    df = pd.read_csv(in_path)
    month, day = map(int, tag.split("_"))
    rows = []
    for _, row in df.iterrows():
        as_of = pd.Timestamp(year=int(row["year"]), month=month, day=day)
        feats = {"field_id": int(row["field_id"]), "year": int(row["year"])}
        feats.update(summarize_management(row, ops, fert, as_of))
        rows.append(feats)
    mgmt = pd.DataFrame(rows)
    out = df.merge(mgmt, on=["field_id", "year"], how="left")
    out.to_csv(out_path, index=False)
    new_cols = [c for c in mgmt.columns if c not in {"field_id", "year"}]
    cov = out[new_cols].notna().mean().sort_values(ascending=False)
    print(f"  {tag}: {len(df)} rows, {df.shape[1]} -> {out.shape[1]} cols")
    print(f"    ops coverage: {out['mgmt_has_ops_asof'].mean()*100:.1f}%")
    print(f"    top feature coverage: {cov.head(10).round(2).to_dict()}")


def main() -> None:
    print("=" * 80)
    print("BUILD v13 — MANAGEMENT FEATURES")
    print("=" * 80)
    ops = load_operations()
    fert = load_fertilizers()
    print(f"Operations: {len(ops)} rows, {ops['field_id'].nunique()} fields")
    print(f"Fertilizers map: {len(fert)} rows")
    for tag in ASOF_TAGS:
        build_one(tag, ops, fert)
    print("\nDone.")


if __name__ == "__main__":
    main()
