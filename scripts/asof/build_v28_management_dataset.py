"""v28 dataset: v27 harvest+S2 + management features from Cropwise operations.

Feature policy:
  - as-of safe: only operations with actual_start_datetime <= as-of date;
  - only real operations: status in {done, in_progress}; canceled/planned are excluded;
  - harvesting is excluded from features to avoid target leakage;
  - material rates use fact_rate if >0, otherwise planned_rate, because many done
    operations have fact_rate=0 while planned_rate carries the recorded dose.

Outputs:
  data_processed/ml_dataset_v28_management_asof_{07_01,08_01,09_01}.csv
  reports/V28_DATASET_SUMMARY.md
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
ASOF_DATES = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}
VALID_STATUS = {"done", "in_progress"}


def safe_items(value: object) -> list[dict]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def num(value: object) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) and np.isfinite(float(out)) else np.nan


def effective_rate(item: dict) -> float:
    fact = num(item.get("fact_rate"))
    planned = num(item.get("planned_rate"))
    rate = fact if pd.notna(fact) and fact > 0 else planned
    return float(rate) if pd.notna(rate) and rate > 0 else 0.0


def load_material_dicts() -> tuple[pd.DataFrame, pd.DataFrame]:
    fert = pd.read_csv(DATA_RAW / "cropwise_fertilizers_full.csv", low_memory=False)
    fert = fert.rename(columns={"id": "applicable_id"})
    for c in ["elements.N", "elements.P2O5", "elements.K2O", "elements.S", "elements.Zn"]:
        if c in fert.columns:
            fert[c] = pd.to_numeric(fert[c], errors="coerce").fillna(0.0) / 100.0
    chem = pd.read_csv(DATA_RAW / "cropwise_chemicals_full.csv", low_memory=False)
    chem = chem.rename(columns={"id": "applicable_id"})
    return fert, chem


def load_ops_long() -> pd.DataFrame:
    ops = pd.read_csv(DATA_RAW / "cropwise_agro_operations_full.csv", low_memory=False)
    ops = ops[ops["status"].isin(VALID_STATUS)].copy()
    ops = ops[~ops["operation_type"].eq("harvesting")].copy()
    ops["field_id"] = pd.to_numeric(ops["field_id"], errors="coerce").astype("Int64")
    ops["year"] = pd.to_numeric(ops["season"], errors="coerce").astype("Int64")
    ops["op_date"] = pd.to_datetime(ops["actual_start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    ops = ops.dropna(subset=["field_id", "year", "op_date"]).copy()
    ops["field_id"] = ops["field_id"].astype(int)
    ops["year"] = ops["year"].astype(int)
    return ops


def material_rows(ops: pd.DataFrame) -> pd.DataFrame:
    fert, chem = load_material_dicts()
    rows = []
    for r in ops.itertuples(index=False):
        for item in safe_items(getattr(r, "application_mix_items", None)):
            typ = item.get("applicable_type")
            rate = effective_rate(item)
            if rate <= 0:
                continue
            rows.append(
                {
                    "field_id": int(r.field_id),
                    "year": int(r.year),
                    "op_date": r.op_date,
                    "operation_type": r.operation_type,
                    "applicable_type": typ,
                    "applicable_id": pd.to_numeric(item.get("applicable_id"), errors="coerce"),
                    "rate": rate,
                    "unit": item.get("rate_unit_label_per_area"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["applicable_id"] = out["applicable_id"].astype("Int64")
    out = out.merge(
        fert[["applicable_id", "elements.N", "elements.P2O5", "elements.K2O", "elements.S", "elements.Zn"]],
        on="applicable_id",
        how="left",
    )
    out = out.merge(chem[["applicable_id", "chemical_type"]], on="applicable_id", how="left", suffixes=("", "_chem"))
    return out


def empty_features() -> dict[str, float]:
    return {
        "mgmt_ops_count_asof": 0,
        "mgmt_application_count_asof": 0,
        "mgmt_soil_count_asof": 0,
        "mgmt_other_count_asof": 0,
        "mgmt_days_since_last_operation_asof": np.nan,
        "mgmt_days_since_last_application_asof": np.nan,
        "mgmt_fertilizer_ops_asof": 0,
        "mgmt_fertilizer_rate_kg_ha_asof": 0.0,
        "mgmt_n_kg_ha_asof": 0.0,
        "mgmt_p2o5_kg_ha_asof": 0.0,
        "mgmt_k2o_kg_ha_asof": 0.0,
        "mgmt_s_kg_ha_asof": 0.0,
        "mgmt_chemical_ops_asof": 0,
        "mgmt_chemical_rate_asof": 0.0,
        "mgmt_herbicide_rate_asof": 0.0,
        "mgmt_fungicide_rate_asof": 0.0,
        "mgmt_insecticide_rate_asof": 0.0,
        "mgmt_seed_rate_asof": 0.0,
    }


def aggregate_one(base: pd.DataFrame, ops: pd.DataFrame, mats: pd.DataFrame, tag: str, month: int, day: int) -> pd.DataFrame:
    ops_by_key = {k: g.sort_values("op_date") for k, g in ops.groupby(["field_id", "year"])}
    mats_by_key = {k: g.sort_values("op_date") for k, g in mats.groupby(["field_id", "year"])} if not mats.empty else {}

    feats = []
    for row in base.itertuples(index=False):
        asof = pd.Timestamp(year=int(row.year), month=month, day=day)
        key = (int(row.field_id), int(row.year))
        rec = empty_features()

        g = ops_by_key.get(key)
        if g is not None:
            past = g[g["op_date"] <= asof]
            rec["mgmt_ops_count_asof"] = int(len(past))
            for typ in ["application", "soil", "other"]:
                rec[f"mgmt_{typ}_count_asof"] = int((past["operation_type"] == typ).sum())
            if len(past):
                rec["mgmt_days_since_last_operation_asof"] = float((asof - past["op_date"].max()).days)
            app = past[past["operation_type"] == "application"]
            if len(app):
                rec["mgmt_days_since_last_application_asof"] = float((asof - app["op_date"].max()).days)

        m = mats_by_key.get(key)
        if m is not None:
            m = m[m["op_date"] <= asof]
            fert = m[m["applicable_type"] == "Fertilizer"]
            rec["mgmt_fertilizer_ops_asof"] = int(len(fert))
            rec["mgmt_fertilizer_rate_kg_ha_asof"] = float(fert["rate"].sum())
            for src, dst in [
                ("elements.N", "mgmt_n_kg_ha_asof"),
                ("elements.P2O5", "mgmt_p2o5_kg_ha_asof"),
                ("elements.K2O", "mgmt_k2o_kg_ha_asof"),
                ("elements.S", "mgmt_s_kg_ha_asof"),
            ]:
                if src in fert:
                    rec[dst] = float((fert["rate"] * fert[src].fillna(0.0)).sum())

            chem = m[m["applicable_type"] == "Chemical"]
            rec["mgmt_chemical_ops_asof"] = int(len(chem))
            rec["mgmt_chemical_rate_asof"] = float(chem["rate"].sum())
            for typ, col in [
                ("herbicide", "mgmt_herbicide_rate_asof"),
                ("fungicide", "mgmt_fungicide_rate_asof"),
                ("insecticide", "mgmt_insecticide_rate_asof"),
            ]:
                rec[col] = float(chem.loc[chem["chemical_type"].eq(typ), "rate"].sum())

            seed = m[m["applicable_type"] == "Seed"]
            rec["mgmt_seed_rate_asof"] = float(seed["rate"].sum())
        feats.append(rec)

    return pd.concat([base.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def main() -> None:
    ops = load_ops_long()
    mats = material_rows(ops)
    print(f"ops={len(ops):,}; material_rows={len(mats):,}")
    summary = []
    for tag, (month, day) in ASOF_DATES.items():
        base = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v27_s2_harvest_asof_{tag}.csv")
        out = aggregate_one(base, ops, mats, tag, month, day)
        path = DATA_PROCESSED / f"ml_dataset_v28_management_asof_{tag}.csv"
        out.to_csv(path, index=False)
        summary.append(
            {
                "asof_tag": tag,
                "rows": len(out),
                "ops_cov": f"{(out['mgmt_ops_count_asof'] > 0).mean():.0%}",
                "fert_cov": f"{(out['mgmt_fertilizer_ops_asof'] > 0).mean():.0%}",
                "chem_cov": f"{(out['mgmt_chemical_ops_asof'] > 0).mean():.0%}",
                "mean_n": round(float(out["mgmt_n_kg_ha_asof"].mean()), 2),
                "mean_chem_rate": round(float(out["mgmt_chemical_rate_asof"].mean()), 2),
            }
        )
        print(f"{tag}: rows={len(out)}, ops_cov={summary[-1]['ops_cov']} -> {path}")

    sm = pd.DataFrame(summary)
    md = ["# v28 dataset summary (management features)\n\n"]
    md.append("As-of safe operation/material aggregates from Cropwise `agro_operations`.\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V28_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
