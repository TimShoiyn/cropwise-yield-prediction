"""
v10 = v9 + internal-only agronomic signals already present in data_raw.

No external API and no Cropwise forecast as a feature.

Adds honest as-of features:
  - scout reports up to as-of date: BBCH/growth_stage, threats, field condition
  - soil test samples up to as-of date: pH, P, K, S, micronutrients, organic matter

Outputs:
  data_processed/ml_dataset_clean_v10_asof_{tag}.csv
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

SOIL_SAMPLE_COLS = [
    "soil_K",
    "soil_P",
    "soil_S",
    "soil_Cu",
    "soil_Fe",
    "soil_Mg",
    "soil_Mn",
    "soil_Mo",
    "soil_Zn",
    "soil_pH",
    "soil_N_NO3",
    "soil_organic_matter",
    "soil_B",
    "soil_N",
]

CONDITION_SCORE = {
    "bad": 0.0,
    "satisfactory": 1.0,
    "good": 2.0,
    "excellent": 3.0,
}


def clean_datetime(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_convert(None)


def parse_threats(value: Any) -> list[dict[str, Any]]:
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
        return [item for item in parsed if isinstance(item, dict)]
    return []


def load_scout() -> pd.DataFrame:
    cols = [
        "field_id",
        "report_time",
        "season",
        "growth_stage",
        "threats",
        "risk_yield_decreasing",
        "field_condition",
    ]
    scout = pd.read_csv(DATA_RAW / "field_scout_reports_aggregated.csv", usecols=cols, low_memory=False)
    scout["field_id"] = pd.to_numeric(scout["field_id"], errors="coerce").astype("Int64")
    scout = scout.dropna(subset=["field_id"]).copy()
    scout["field_id"] = scout["field_id"].astype(int)
    scout["report_time"] = clean_datetime(scout["report_time"])
    scout["season"] = pd.to_numeric(scout["season"], errors="coerce").astype("Int64")
    scout["growth_stage_num"] = pd.to_numeric(scout["growth_stage"], errors="coerce")
    scout["risk_yield_decreasing"] = scout["risk_yield_decreasing"].astype(str).str.lower().eq("true")
    cond = scout["field_condition"].astype(str).str.lower().replace({"nan": np.nan})
    scout["field_condition_score"] = cond.map(CONDITION_SCORE)
    scout["condition_bad"] = cond.eq("bad")
    scout["condition_not_good"] = cond.isin(["bad", "satisfactory"])

    parsed = scout["threats"].apply(parse_threats)
    scout["threat_count"] = parsed.apply(len)
    scout["threat_has_disease"] = parsed.apply(lambda items: any(str(x.get("type", "")).lower() == "disease" for x in items))
    scout["threat_has_insect"] = parsed.apply(lambda items: any(str(x.get("type", "")).lower() == "insect" for x in items))
    scout["threat_has_weed"] = parsed.apply(lambda items: any(str(x.get("type", "")).lower() == "weed" for x in items))
    return scout


def load_soil_samples() -> pd.DataFrame:
    tests = pd.read_csv(DATA_RAW / "soil_tests.csv", usecols=["id", "field_id", "made_at"], low_memory=False)
    tests = tests.rename(columns={"id": "soil_test_id"})
    tests["field_id"] = pd.to_numeric(tests["field_id"], errors="coerce").astype("Int64")
    tests["made_at"] = pd.to_datetime(tests["made_at"], errors="coerce")
    samples = pd.read_csv(
        DATA_RAW / "soil_test_samples.csv",
        usecols=["soil_test_id"] + SOIL_SAMPLE_COLS,
        low_memory=False,
    )
    samples = samples.merge(tests, on="soil_test_id", how="left")
    samples = samples.dropna(subset=["field_id"]).copy()
    samples["field_id"] = samples["field_id"].astype(int)
    for col in SOIL_SAMPLE_COLS:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    return samples


def scout_features_for_row(row: pd.Series, scout: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float | int]:
    fid = int(row["field_id"])
    year = int(row["year"])
    g = scout[
        (scout["field_id"] == fid)
        & (scout["season"] == year)
        & scout["report_time"].notna()
        & (scout["report_time"] <= as_of)
    ].sort_values("report_time")

    out: dict[str, float | int] = {
        "scout_n_reports_asof": int(len(g)),
        "scout_has_report_asof": int(len(g) > 0),
    }
    if g.empty:
        return out

    out["scout_last_report_doy"] = int(g["report_time"].iloc[-1].dayofyear)
    out["scout_days_since_last_report"] = int((as_of - g["report_time"].iloc[-1]).days)
    out["scout_growth_stage_max"] = float(g["growth_stage_num"].max())
    out["scout_growth_stage_last"] = float(g["growth_stage_num"].dropna().iloc[-1]) if g["growth_stage_num"].notna().any() else np.nan
    out["scout_risk_yield_decreasing_n"] = int(g["risk_yield_decreasing"].sum())
    out["scout_risk_yield_decreasing_any"] = int(g["risk_yield_decreasing"].any())
    out["scout_condition_score_min"] = float(g["field_condition_score"].min())
    out["scout_condition_score_last"] = float(g["field_condition_score"].dropna().iloc[-1]) if g["field_condition_score"].notna().any() else np.nan
    out["scout_condition_bad_n"] = int(g["condition_bad"].sum())
    out["scout_condition_not_good_n"] = int(g["condition_not_good"].sum())
    out["scout_threat_reports_n"] = int((g["threat_count"] > 0).sum())
    out["scout_threat_total_n"] = int(g["threat_count"].sum())
    out["scout_threat_disease_any"] = int(g["threat_has_disease"].any())
    out["scout_threat_insect_any"] = int(g["threat_has_insect"].any())
    out["scout_threat_weed_any"] = int(g["threat_has_weed"].any())
    return out


def soil_features_for_row(row: pd.Series, soil: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float | int]:
    fid = int(row["field_id"])
    g = soil[
        (soil["field_id"] == fid)
        & soil["made_at"].notna()
        & (soil["made_at"] <= as_of)
    ].copy()
    out: dict[str, float | int] = {
        "soil_sample_n_asof": int(len(g)),
        "soil_sample_has_asof": int(len(g) > 0),
    }
    if g.empty:
        return out

    latest_date = g["made_at"].max()
    latest = g[g["made_at"] == latest_date]
    out["soil_sample_days_since_latest"] = int((as_of - latest_date).days)
    out["soil_sample_n_tests_asof"] = int(g["soil_test_id"].nunique())
    for col in SOIL_SAMPLE_COLS:
        vals = pd.to_numeric(latest[col], errors="coerce")
        out[f"soil_sample_latest_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
        all_vals = pd.to_numeric(g[col], errors="coerce")
        out[f"soil_sample_mean_{col}"] = float(all_vals.mean()) if all_vals.notna().any() else np.nan
    return out


def build_one(tag: str, scout: pd.DataFrame, soil: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v9_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v10_asof_{tag}.csv"
    df = pd.read_csv(in_path)
    month, day = map(int, tag.split("_"))

    rows = []
    for _, row in df.iterrows():
        as_of = pd.Timestamp(year=int(row["year"]), month=month, day=day)
        feats = {"field_id": int(row["field_id"]), "year": int(row["year"])}
        feats.update(scout_features_for_row(row, scout, as_of))
        feats.update(soil_features_for_row(row, soil, as_of))
        rows.append(feats)

    features = pd.DataFrame(rows)
    out = df.merge(features, on=["field_id", "year"], how="left")
    out.to_csv(out_path, index=False)

    new_cols = [c for c in features.columns if c not in {"field_id", "year"}]
    coverage = out[new_cols].notna().mean().sort_values(ascending=False)
    print(f"  {tag}: {len(df)} rows, {df.shape[1]} -> {out.shape[1]} cols")
    print(f"    scout rows with report: {out['scout_has_report_asof'].mean() * 100:.1f}%")
    print(f"    soil rows with sample:  {out['soil_sample_has_asof'].mean() * 100:.1f}%")
    print(f"    top coverage: {coverage.head(8).round(2).to_dict()}")


def main() -> None:
    print("=" * 80)
    print("BUILD v10 — INTERNAL SCOUT + SOIL SAMPLE FEATURES")
    print("=" * 80)
    scout = load_scout()
    soil = load_soil_samples()
    print(f"Scout reports: {len(scout)} rows, {scout['field_id'].nunique()} fields")
    print(f"Soil samples:  {len(soil)} rows, {soil['field_id'].nunique()} fields")
    for tag in ASOF_TAGS:
        build_one(tag, scout, soil)
    print("\nDone.")


if __name__ == "__main__":
    main()
