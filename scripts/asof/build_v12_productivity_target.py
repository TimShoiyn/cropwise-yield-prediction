"""
v12: use productivity_data.csv as target source for exact-matched VKO fields.

Why:
  productivity_data.csv has field name + fact yield in c/ha. It revealed that
  targets_factual_t_ha still had 10x unit errors for low-yield 2020 rows.

Outputs:
  data_processed/ml_dataset_clean_v12_asof_{tag}.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]


def norm(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def productivity_target_table() -> pd.DataFrame:
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "name"]].rename(columns={"id": "field_id", "name": "field_name"})
    fields["field_name_norm"] = fields["field_name"].map(norm)
    prod = pd.read_csv(DATA_RAW / "productivity_data.csv")
    prod["field_name_norm"] = prod["Поле"].map(norm)
    prod = prod.rename(columns={"Год": "year", "Культура": "prod_crop_ru"})
    prod["prod_fact_t_ha"] = pd.to_numeric(prod["урожайность факт ц/га"], errors="coerce") / 10.0
    prod["prod_forecast_t_ha"] = pd.to_numeric(prod["урожайность прогноз ц/га"], errors="coerce") / 10.0
    out = prod.merge(fields, on="field_name_norm", how="inner")
    out = out.dropna(subset=["prod_fact_t_ha"]).copy()
    out["field_id"] = out["field_id"].astype(int)
    out["year"] = out["year"].astype(int)
    # Do not keep `prod_forecast_t_ha` in the model dataset: it is Cropwise's
    # own forecast and must remain an external benchmark only.
    return out[["field_id", "year", "field_name", "prod_crop_ru", "prod_fact_t_ha"]]


def build_one(tag: str, prod_target: pd.DataFrame) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v10_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v12_asof_{tag}.csv"
    df = pd.read_csv(in_path)
    if "standard_name" not in df.columns and "crop_id" in df.columns:
        crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
        df = df.merge(crops, on="crop_id", how="left")
    old = df[["field_id", "year", "target_yield_t_ha", "target_source"]].rename(
        columns={"target_yield_t_ha": "old_target_yield_t_ha", "target_source": "old_target_source"}
    )
    df = df.drop(columns=["target_yield_t_ha", "target_source", "unit_fix_applied"], errors="ignore")
    out = df.merge(prod_target, on=["field_id", "year"], how="inner")
    out["target_yield_t_ha"] = out["prod_fact_t_ha"]
    out["target_source"] = "productivity_data_c_ha_exact_field_name"
    out["unit_fix_applied"] = True
    # Keep training file clean: no old target, no target deltas, no Cropwise
    # forecast from productivity_data.
    out = out.drop(columns=["prod_fact_t_ha"], errors="ignore")
    out.to_csv(out_path, index=False)
    changed = len(old.merge(prod_target, on=["field_id", "year"], how="inner").query("abs(prod_fact_t_ha - old_target_yield_t_ha) > 0.1"))
    print(f"  {tag}: {len(df)} -> {len(out)} rows, changed target >0.1 t/ha: {changed}")


def main() -> None:
    print("=" * 80)
    print("BUILD v12 — PRODUCTIVITY_DATA TARGET OVERRIDE")
    print("=" * 80)
    prod_target = productivity_target_table()
    print(f"Productivity target rows exact-matched: {len(prod_target)}")
    print(f"Fields: {prod_target['field_id'].nunique()}, years: {prod_target['year'].min()}..{prod_target['year'].max()}")
    for tag in ASOF_TAGS:
        build_one(tag, prod_target)
    print("\nDone.")


if __name__ == "__main__":
    main()
