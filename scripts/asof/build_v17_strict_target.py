"""
Build v17 strict-target datasets.

For every (field_id, year) we recompute the canonical target with the
priority:
    physical_t_ha (= harvested_weight / tillable_area)
        > prod_fact_t_ha  (productivity_data c/ha -> t/ha via field-name match)
        > productivity_t_ha_norm (history_items.productivity, c/ha -> t/ha
          per-crop cap normalization; same as v9 baseline).

Any row where TWO independent sources disagree by more than the threshold
(1.0 t/ha by default) is treated as a target conflict and marked
`target_conflict=True`. We emit two flavours:
    v17_full      : every row, target = best-of-priority, conflict-flag kept;
    v17_clean     : conflict rows dropped (strict).

The script applies this transform to every existing v12 / v14 / v15_rates
asof CSV in `data_processed/`, generating sibling files
`ml_dataset_clean_v17_<base>_full_asof_<date>.csv`
`ml_dataset_clean_v17_<base>_clean_asof_<date>.csv`.

Outputs:
    data_processed/ml_dataset_clean_v17_*_full_asof_*.csv
    data_processed/ml_dataset_clean_v17_*_clean_asof_*.csv
    reports/V17_TARGET_REBUILD.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")

CONFLICT_THRESHOLD_T_HA = 1.0
ASOF_TAGS = ["07_01", "08_01", "09_01"]
BASES = ["v12", "v14", "v15_rates"]

MAX_T_HA = {
    "sunflower": 5.0,
    "wheat_spring": 7.0,
    "wheat_winter": 8.0,
    "barley_spring": 7.0,
    "maize": 12.0,
    "oil_seed_raps_spring": 4.5,
    "oil_seed_raps_winter": 5.0,
    "soya": 4.0,
    "pea": 4.5,
    "lentil": 3.5,
    "buckwheat": 3.0,
    "safflower": 2.5,
    "fallow": 2.0,
    "linum": 3.0,
    "avena_spring": 4.5,
    "rye_winter": 6.0,
    "sainfoin": 5.0,
    "medicago": 5.0,
    "sudan_grass": 6.0,
    "sweet_clover": 5.0,
    "potatoes": 60.0,
    "sugar_beet": 80.0,
}
DEFAULT_MAX_T_HA = 5.0


def normalize_yield(raw: float, std_name: str | None) -> float:
    if pd.isna(raw):
        return np.nan
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    value = float(raw)
    if value > cap:
        return value / 10.0
    return value


def _norm(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_target_table() -> pd.DataFrame:
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "name", "tillable_area"]].rename(columns={"id": "field_id", "name": "field_name"})
    fields["field_name_norm"] = fields["field_name"].map(_norm)

    hi = pd.read_csv(
        DATA_RAW / "history_items_full.csv",
        low_memory=False,
        usecols=["field_id", "year", "crop_id", "productivity", "harvested_weight"],
    )
    hi["field_id"] = pd.to_numeric(hi["field_id"], errors="coerce").astype("Int64")
    hi["year"] = pd.to_numeric(hi["year"], errors="coerce").astype("Int64")
    hi = hi.dropna(subset=["field_id", "year"]).copy()
    hi["field_id"] = hi["field_id"].astype(int)
    hi["year"] = hi["year"].astype(int)
    hi = hi.merge(crops, on="crop_id", how="left")
    hi = hi.merge(fields[["field_id", "tillable_area", "field_name_norm"]], on="field_id", how="left")

    hi["physical_t_ha_raw"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(hi["tillable_area"], errors="coerce")
    hi["physical_t_ha"] = [normalize_yield(v, c) for v, c in zip(hi["physical_t_ha_raw"], hi["standard_name"])]
    hi["productivity_t_ha"] = [normalize_yield(v, c) for v, c in zip(pd.to_numeric(hi["productivity"], errors="coerce"), hi["standard_name"])]

    prod = pd.read_csv(DATA_RAW / "productivity_data.csv")
    prod["field_name_norm"] = prod["Поле"].map(_norm)
    prod = prod.rename(columns={"Год": "year"})
    prod["year"] = pd.to_numeric(prod["year"], errors="coerce").astype("Int64")
    prod = prod.dropna(subset=["year"]).copy()
    prod["year"] = prod["year"].astype(int)
    prod["prod_fact_t_ha"] = pd.to_numeric(prod["урожайность факт ц/га"], errors="coerce") / 10.0
    prod = prod.dropna(subset=["prod_fact_t_ha"]).copy()
    prod_join = prod.merge(fields[["field_id", "field_name_norm"]], on="field_name_norm", how="inner")
    prod_join = prod_join[["field_id", "year", "prod_fact_t_ha"]].drop_duplicates(["field_id", "year"])

    grouped = hi.groupby(["field_id", "year"], as_index=False).agg(
        physical_t_ha=("physical_t_ha", "max"),
        productivity_t_ha=("productivity_t_ha", "max"),
    )
    grouped = grouped.merge(prod_join, on=["field_id", "year"], how="left")

    sources = ["physical_t_ha", "prod_fact_t_ha", "productivity_t_ha"]

    def pick_target(row: pd.Series):
        for col in sources:
            value = row.get(col)
            if pd.notna(value):
                return value, col
        return np.nan, None

    grouped["target_v17"], grouped["target_v17_source"] = zip(*grouped.apply(pick_target, axis=1))

    def detect_conflict(row: pd.Series) -> bool:
        # Compare only the two trusted sources (physical and prod_fact).
        # productivity_t_ha is used only as a last-resort fallback and is
        # known to mix c/ha with t/ha; including it explodes the conflict
        # set with non-meaningful disagreements.
        a = row.get("physical_t_ha")
        b = row.get("prod_fact_t_ha")
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(float(a) - float(b)) > CONFLICT_THRESHOLD_T_HA

    grouped["target_v17_conflict"] = grouped.apply(detect_conflict, axis=1)
    return grouped


def transform(in_path: Path, target_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(in_path)
    df = df.merge(target_table, on=["field_id", "year"], how="left")

    matched = df["target_v17"].notna()
    df.loc[matched, "target_yield_t_ha"] = df.loc[matched, "target_v17"]
    df.loc[matched, "target_source"] = "v17_" + df.loc[matched, "target_v17_source"].astype(str)

    info = {
        "rows_in": int(len(df)),
        "rows_v17_target": int(matched.sum()),
        "rows_conflict": int(df["target_v17_conflict"].fillna(False).sum()),
    }
    return df, info


def main() -> None:
    target_table = build_target_table()
    print("target table size:", len(target_table))
    print("conflict rows in target table:", int(target_table["target_v17_conflict"].sum()))

    md = ["# v17 strict-target rebuild\n\n"]
    md.append(f"Conflict threshold: {CONFLICT_THRESHOLD_T_HA} t/ha (max-min across non-null sources).\n")
    md.append("Target priority: physical_t_ha > prod_fact_t_ha > productivity_t_ha (per-crop cap normalized).\n\n")
    md.append("| base | asof | rows_in | rows_v17_target | rows_conflict | rows_clean |\n")
    md.append("|------|------|--------:|----------------:|--------------:|-----------:|\n")

    for base in BASES:
        for tag in ASOF_TAGS:
            in_path = DATA_PROCESSED / f"ml_dataset_clean_{base}_asof_{tag}.csv"
            if not in_path.exists():
                continue
            df, info = transform(in_path, target_table)
            full = df.copy()
            full = full.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
            clean = full[~full["target_v17_conflict"].fillna(False)].reset_index(drop=True)

            full_path = DATA_PROCESSED / f"ml_dataset_clean_v17_{base}_full_asof_{tag}.csv"
            clean_path = DATA_PROCESSED / f"ml_dataset_clean_v17_{base}_clean_asof_{tag}.csv"
            full.to_csv(full_path, index=False)
            clean.to_csv(clean_path, index=False)
            md.append(f"| {base} | {tag} | {info['rows_in']} | {info['rows_v17_target']} | {info['rows_conflict']} | {len(clean)} |\n")
            print(f"{base} {tag}: in={info['rows_in']} v17_target={info['rows_v17_target']} conflict={info['rows_conflict']} clean={len(clean)}")

    out = REPORTS / "V17_TARGET_REBUILD.md"
    out.write_text("".join(md), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
