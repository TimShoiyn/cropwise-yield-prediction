"""
High-yield miss audit for v12.

Focus: rows where target is high but ML strongly underpredicts.
Checks whether the current model had NDVI/weather/soil/scout signals and
whether operations table contains management signals not present as features.

Output: reports/V12_HIGH_YIELD_MISS_AUDIT.md
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
REPORTS = Path("reports")

SELECTED_FEATURES = [
    "field_tillable_area",
    "field_soil_pH",
    "field_soil_OM",
    "field_soil_P",
    "field_soil_K",
    "field_soil_N",
    "field_soil_Mg",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_last_value_asof",
    "ndvi_integral_asof",
    "ndvi_peak_doy_asof",
    "ndvi_anom_mean_asof",
    "ndvi_anom_last_value_asof",
    "wx_precip_sum_to_asof",
    "wx_et0_sum_to_asof",
    "wx_water_balance_to_asof",
    "wx_vpd_max_to_asof",
    "wx_hot_d30_to_asof",
    "wx_dry_days_to_asof",
    "wx_sow_precip_sum",
    "wx_sow_water_balance",
    "sowing_date_valid",
    "days_after_sowing_asof",
    "gdd_from_sowing_asof",
    "scout_n_reports_asof",
    "scout_risk_yield_decreasing_any",
    "scout_condition_bad_n",
    "scout_threat_total_n",
    "soil_sample_has_asof",
    "soil_sample_latest_soil_pH",
    "soil_sample_latest_soil_P",
    "soil_sample_latest_soil_K",
    "soil_sample_latest_soil_N_NO3",
    "soil_sample_latest_soil_organic_matter",
]


def code(df_or_text) -> str:
    if isinstance(df_or_text, pd.DataFrame):
        text = df_or_text.to_string(index=False)
    elif isinstance(df_or_text, pd.Series):
        text = df_or_text.to_string()
    else:
        text = str(df_or_text)
    return f"\n```\n{text}\n```\n"


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


def operation_summary(ops: pd.DataFrame, field_id: int, year: int, asof_tag: str) -> dict[str, Any]:
    month, day = map(int, asof_tag.split("_"))
    asof = pd.Timestamp(year=year, month=month, day=day)
    g = ops[(ops["field_id"] == field_id) & (ops["season"] == year)].copy()
    if g.empty:
        return {"ops_n_asof": 0}
    g["op_date"] = pd.to_datetime(g["actual_start_datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    g["op_date"] = g["op_date"].fillna(pd.to_datetime(g["completed_date"], errors="coerce"))
    g = g[g["op_date"].notna() & (g["op_date"] <= asof)].copy()
    if g.empty:
        return {"ops_n_asof": 0}

    mix_items = []
    for _, row in g.iterrows():
        for item in parse_mix(row.get("application_mix_items")):
            item = dict(item)
            item["operation_id"] = row.get("id")
            item["operation_type"] = row.get("operation_type")
            item["op_date"] = row.get("op_date")
            mix_items.append(item)
    mix_df = pd.DataFrame(mix_items)

    out: dict[str, Any] = {
        "ops_n_asof": int(len(g)),
        "ops_completed_area_sum": float(pd.to_numeric(g["completed_area"], errors="coerce").sum()),
        "ops_application_n": int((g["operation_type"] == "application").sum()),
        "ops_sowing_n": int((g["operation_type"] == "sowing").sum()),
        "ops_harvesting_n_asof": int((g["operation_type"] == "harvesting").sum()),
        "ops_type_list": ";".join(g["operation_type"].fillna("NA").astype(str).value_counts().index[:8]),
        "ops_last_date": str(g["op_date"].max().date()),
    }
    if not mix_df.empty:
        mix_df["fact_rate"] = pd.to_numeric(mix_df.get("fact_rate"), errors="coerce")
        mix_df["fact_amount"] = pd.to_numeric(mix_df.get("fact_amount"), errors="coerce")
        for applicable_type in ["Fertilizer", "Seed", "Chemical"]:
            sub = mix_df[mix_df.get("applicable_type").astype(str).eq(applicable_type)] if "applicable_type" in mix_df else pd.DataFrame()
            out[f"mix_{applicable_type.lower()}_n"] = int(len(sub))
            out[f"mix_{applicable_type.lower()}_rate_sum"] = float(sub["fact_rate"].sum()) if not sub.empty and "fact_rate" in sub else 0.0
        out["mix_applicable_types"] = ";".join(mix_df.get("applicable_type", pd.Series(dtype=str)).fillna("NA").astype(str).value_counts().index[:8])
    return out


def yield_map_summary(field_id: int, year: int) -> dict[str, Any]:
    ym = pd.read_csv(DATA_RAW / "yield_maps.csv", low_memory=False)
    ym["created_at"] = pd.to_datetime(ym["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    ym["year"] = ym["created_at"].dt.year
    g = ym[(ym["field_id"] == field_id) & (ym["year"] == year)].copy()
    if g.empty:
        return {"yield_map_n": 0}
    vals = []
    for _, row in g.iterrows():
        val = pd.to_numeric(row.get("external_average"), errors="coerce")
        if pd.isna(val):
            val = pd.to_numeric(row.get("totals.result.average.value"), errors="coerce")
        if pd.notna(val) and 0.1 <= float(val) <= 12:
            vals.append(float(val))
    return {
        "yield_map_n": int(len(g)),
        "yield_map_mean": float(np.mean(vals)) if vals else np.nan,
        "yield_map_values": ";".join(f"{v:.2f}" for v in vals[:10]),
    }


def main() -> None:
    preds = pd.read_csv(REPORTS / "v12_oof_predictions_best.csv")
    allc = preds[preds["scenario"] == "all_crops"].copy()
    allc["underprediction"] = allc["target_yield_t_ha"] - allc["ml_pred"]
    top = allc.sort_values("underprediction", ascending=False).head(25).copy()

    ops = pd.read_csv(
        DATA_RAW / "operations.csv",
        usecols=[
            "id", "field_id", "season", "operation_type", "applications_type", "completed_area",
            "actual_start_datetime", "completed_date", "application_mix_items",
        ],
        low_memory=False,
    )

    rows = []
    for _, r in top.iterrows():
        tag = r["asof_tag"]
        feature_df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v12_asof_{tag}.csv")
        frow = feature_df[(feature_df["field_id"] == r["field_id"]) & (feature_df["year"] == r["year"])]
        feat = {}
        if not frow.empty:
            for col in SELECTED_FEATURES:
                if col in frow.columns:
                    feat[col] = frow.iloc[0][col]
        op = operation_summary(ops, int(r["field_id"]), int(r["year"]), tag)
        ym = yield_map_summary(int(r["field_id"]), int(r["year"]))
        rows.append({**r.to_dict(), **feat, **op, **ym})

    audit = pd.DataFrame(rows)
    out_csv = REPORTS / "v12_high_yield_miss_audit.csv"
    audit.to_csv(out_csv, index=False)

    md: list[str] = ["# v12 High-Yield Miss Audit\n\n"]
    md.append("Generated by `scripts/audit/high_yield_miss_audit_v12.py`.\n\n")
    md.append("Focus: rows where target is high and ML underpredicts strongly.\n\n")
    md.append(f"Detailed CSV: `{out_csv}`.\n\n")

    md.append("## 1. Top high-yield underpredictions\n")
    show_cols = [
        "asof_tag", "field_id", "field_name_csv", "year", "standard_name", "target_yield_t_ha",
        "ml_pred", "cropwise_asof_t_ha", "underprediction", "ml_abs_error", "cw_abs_error",
        "ndvi_max_asof", "ndvi_integral_asof", "wx_water_balance_to_asof", "scout_n_reports_asof",
        "soil_sample_has_asof", "ops_n_asof", "mix_fertilizer_n", "mix_seed_n", "yield_map_mean",
    ]
    present_cols = [c for c in show_cols if c in audit.columns]
    md.append(code(audit[present_cols].round(3)))

    md.append("\n## 2. Operation signal presence\n")
    op_cols = [c for c in [
        "field_id", "year", "asof_tag", "ops_n_asof", "ops_application_n", "ops_sowing_n",
        "mix_fertilizer_n", "mix_fertilizer_rate_sum", "mix_seed_n", "mix_seed_rate_sum",
        "mix_chemical_n", "mix_applicable_types", "ops_type_list",
    ] if c in audit.columns]
    md.append(code(audit[op_cols].round(3)))

    md.append("\n## 3. Interpretation\n\n")
    md.append("- Current v12 feature set has no explicit operation/NPK/seed-rate feature columns.\n")
    md.append("- The operations table does contain fertilizer/seed/chemical events for several high-yield misses.\n")
    md.append("- Therefore v13 should focus on management features before trying another model family.\n")
    md.append("- Also inspect rows where `yield_map_mean` exists: if yield map agrees with target, these are real high-yield cases, not label noise.\n")

    out_md = REPORTS / "V12_HIGH_YIELD_MISS_AUDIT.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(audit[present_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
