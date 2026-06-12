"""
v23 dataset builder: full Cropwise historical_values on all fields.

This is the first local dataset after Alpamys's refreshed token unlocked:
  - 640 field geometries;
  - 686 soil tests;
  - historical_values for all 633 fields that have yield history.

Target policy matches v17:
  physical harvested_weight / area > productivity_data fact > history productivity.
Conflicts between physical and productivity_data fact (>1 t/ha) are dropped.

Outputs:
  data_processed/ml_dataset_v23_cropwise_full_asof_{07_01,08_01,09_01}.csv
  reports/V23_DATASET_SUMMARY.md
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
DATA_PROCESSED.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

ASOF_DATES = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}

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


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_yield(raw: object, standard_name: object) -> float:
    value = pd.to_numeric(raw, errors="coerce")
    if pd.isna(value):
        return np.nan
    cap = MAX_T_HA.get(str(standard_name), DEFAULT_MAX_T_HA)
    value = float(value)
    return value / 10.0 if value > cap else value


def load_base() -> pd.DataFrame:
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    fields = pd.read_csv(DATA_RAW / "cropwise_fields_full.csv").rename(columns={"id": "field_id", "name": "field_name"})
    field_cols = [
        "field_id",
        "field_name",
        "tillable_area",
        "calculated_area",
        "lat",
        "long",
        "region_id",
        "district_id",
        "shape_simplified_geojson",
    ]
    fields = fields[[c for c in field_cols if c in fields.columns]].copy()
    fields["field_name_norm"] = fields["field_name"].map(norm)

    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False)
    hi = hi.rename(columns={"id": "history_item_id"})
    hi["field_id"] = pd.to_numeric(hi["field_id"], errors="coerce").astype("Int64")
    hi["year"] = pd.to_numeric(hi["year"], errors="coerce").astype("Int64")
    hi = hi.dropna(subset=["field_id", "year"]).copy()
    hi["field_id"] = hi["field_id"].astype(int)
    hi["year"] = hi["year"].astype(int)
    hi = hi[(hi["year"] >= 2002) & (hi["year"] <= 2026)].copy()
    hi = hi.merge(crops, on="crop_id", how="left")
    hi = hi.merge(fields, on="field_id", how="left")

    hi["physical_t_ha_raw"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(
        hi["tillable_area"], errors="coerce"
    )
    hi["physical_t_ha"] = [
        normalize_yield(v, c) for v, c in zip(hi["physical_t_ha_raw"], hi["standard_name"])
    ]
    hi["productivity_t_ha"] = [
        normalize_yield(v, c) for v, c in zip(hi["productivity"], hi["standard_name"])
    ]

    prod = pd.read_csv(DATA_RAW / "productivity_data.csv")
    prod["field_name_norm"] = prod["Поле"].map(norm)
    prod = prod.rename(columns={"Год": "year"})
    prod["year"] = pd.to_numeric(prod["year"], errors="coerce").astype("Int64")
    prod = prod.dropna(subset=["year"]).copy()
    prod["year"] = prod["year"].astype(int)
    prod["prod_fact_t_ha"] = pd.to_numeric(prod["урожайность факт ц/га"], errors="coerce") / 10.0
    prod_join = prod.merge(fields[["field_id", "field_name_norm"]], on="field_name_norm", how="inner")
    prod_join = prod_join[["field_id", "year", "prod_fact_t_ha"]].drop_duplicates(["field_id", "year"])

    hi = hi.merge(prod_join, on=["field_id", "year"], how="left")

    sources = ["physical_t_ha", "prod_fact_t_ha", "productivity_t_ha"]
    hi["target_yield_t_ha"] = np.nan
    hi["target_source"] = None
    for col in sources:
        mask = hi["target_yield_t_ha"].isna() & hi[col].notna()
        hi.loc[mask, "target_yield_t_ha"] = hi.loc[mask, col]
        hi.loc[mask, "target_source"] = f"v23_{col}"

    hi["target_conflict"] = (
        hi["physical_t_ha"].notna()
        & hi["prod_fact_t_ha"].notna()
        & ((hi["physical_t_ha"] - hi["prod_fact_t_ha"]).abs() > 1.0)
    )
    hi = hi.dropna(subset=["target_yield_t_ha", "standard_name"]).copy()
    hi = hi[~hi["target_conflict"]].copy()

    keep = [
        "history_item_id",
        "field_id",
        "year",
        "crop_id",
        "standard_name",
        "variety",
        "sowing_date",
        "harvesting_date",
        "till_type",
        "target_yield_t_ha",
        "target_source",
        "tillable_area",
        "calculated_area",
        "lat",
        "long",
        "region_id",
        "district_id",
        "field_name",
        "shape_simplified_geojson",
    ]
    out = hi[[c for c in keep if c in hi.columns]].copy()
    out = out.rename(columns={"tillable_area": "field_tillable_area", "calculated_area": "field_calculated_area"})
    return add_rotation_features(out)


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["field_id", "year"]).copy()
    df["prev_crop_id"] = df.groupby("field_id")["crop_id"].shift(1)
    df["prev_standard_name"] = df.groupby("field_id")["standard_name"].shift(1).fillna("missing")
    df["rotation_pair"] = df["prev_standard_name"].astype(str) + "->" + df["standard_name"].astype(str)

    for crop, col in [("sunflower", "years_since_sunflower"), ("wheat", "years_since_wheat")]:
        vals = []
        for _, g in df.groupby("field_id", sort=False):
            last_year = None
            for _, row in g.iterrows():
                vals.append(np.nan if last_year is None else int(row["year"]) - last_year)
                name = str(row["standard_name"])
                if (crop == "wheat" and "wheat" in name) or name == crop:
                    last_year = int(row["year"])
        df[col] = vals
    return df


def parse_series(value: object) -> list[tuple[pd.Timestamp, float]]:
    if pd.isna(value):
        return []
    try:
        raw = json.loads(str(value))
    except Exception:
        return []
    out = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        date = pd.to_datetime(item[0], errors="coerce")
        try:
            val = float(item[1])
        except (TypeError, ValueError):
            continue
        if pd.notna(date) and np.isfinite(val):
            out.append((pd.Timestamp(date), val))
    out.sort(key=lambda x: x[0])
    return out


def load_satellite(base: pd.DataFrame) -> dict[tuple[int, int, str], list[tuple[pd.Timestamp, float]]]:
    needed = set(map(tuple, base[["field_id", "year"]].astype(int).drop_duplicates().values.tolist()))
    sat = pd.read_csv(DATA_RAW / "cropwise_historical_values_full.csv")
    sat = sat[sat["product_type"].isin(["ndvi", "temperature", "soil_moisture"])].copy()
    sat["field_id"] = sat["field_id"].astype(int)
    sat["year"] = sat["year"].astype(int)
    sat = sat[[key in needed for key in zip(sat["field_id"], sat["year"])]].copy()
    print(f"Satellite rows to parse: {len(sat):,} for {len(needed):,} target field-years")
    mapping = {}
    for i, row in enumerate(sat.itertuples(index=False), start=1):
        mapping[(int(row.field_id), int(row.year), str(row.product_type))] = parse_series(row.value_json)
        if i % 1000 == 0:
            print(f"  parsed satellite rows: {i:,}/{len(sat):,}")
    return mapping


def series_features(
    values: list[tuple[pd.Timestamp, float]],
    year: int,
    month: int,
    day: int,
    prefix: str,
) -> dict[str, float]:
    asof = pd.Timestamp(year=int(year), month=month, day=day)
    vals = [(d, v) for d, v in values if d.year == int(year) and d <= asof]
    empty = {
        f"{prefix}_n_obs_asof": 0,
        f"{prefix}_mean_asof": np.nan,
        f"{prefix}_max_asof": np.nan,
        f"{prefix}_min_asof": np.nan,
        f"{prefix}_std_asof": np.nan,
        f"{prefix}_p25_asof": np.nan,
        f"{prefix}_p75_asof": np.nan,
        f"{prefix}_last_value_asof": np.nan,
        f"{prefix}_last_doy_asof": np.nan,
        f"{prefix}_peak_doy_asof": np.nan,
        f"{prefix}_amplitude_asof": np.nan,
        f"{prefix}_slope_asof": np.nan,
        f"{prefix}_integral_asof": np.nan,
        f"{prefix}_last30d_mean_asof": np.nan,
    }
    if not vals:
        return empty

    dates = np.array([d for d, _ in vals])
    arr = np.array([v for _, v in vals], dtype=float)
    doy = np.array([pd.Timestamp(d).dayofyear for d in dates], dtype=float)
    peak_idx = int(np.nanargmax(arr))
    slope = np.nan
    if len(arr) >= 2 and np.nanstd(doy) > 0:
        slope = float(np.polyfit(doy, arr, deg=1)[0])
    last30 = np.array([v for d, v in vals if d >= asof - pd.Timedelta(days=30)], dtype=float)

    out = {
        f"{prefix}_n_obs_asof": int(len(arr)),
        f"{prefix}_mean_asof": float(np.nanmean(arr)),
        f"{prefix}_max_asof": float(np.nanmax(arr)),
        f"{prefix}_min_asof": float(np.nanmin(arr)),
        f"{prefix}_std_asof": float(np.nanstd(arr)),
        f"{prefix}_p25_asof": float(np.nanpercentile(arr, 25)),
        f"{prefix}_p75_asof": float(np.nanpercentile(arr, 75)),
        f"{prefix}_last_value_asof": float(arr[-1]),
        f"{prefix}_last_doy_asof": float(doy[-1]),
        f"{prefix}_peak_doy_asof": float(doy[peak_idx]),
        f"{prefix}_amplitude_asof": float(np.nanmax(arr) - np.nanmin(arr)),
        f"{prefix}_slope_asof": slope,
        f"{prefix}_integral_asof": float(np.trapezoid(arr, doy)) if len(arr) >= 2 else 0.0,
        f"{prefix}_last30d_mean_asof": float(np.nanmean(last30)) if len(last30) else np.nan,
    }
    if prefix == "ndvi":
        out["ndvi_above_05_days_asof"] = int(np.sum(arr >= 0.5))
    return out


def sowing_features(sowing_date: object, year: int, month: int, day: int) -> dict[str, float]:
    asof = pd.Timestamp(year=int(year), month=month, day=day)
    sow = pd.to_datetime(sowing_date, errors="coerce")
    if pd.isna(sow) or sow > asof:
        return {"sowing_known_asof": 0, "sowing_doy": np.nan, "days_after_sowing_asof": np.nan}
    return {
        "sowing_known_asof": 1,
        "sowing_doy": float(pd.Timestamp(sow).dayofyear),
        "days_after_sowing_asof": float((asof - pd.Timestamp(sow)).days),
    }


def load_soil_long() -> pd.DataFrame:
    soil = pd.read_csv(DATA_RAW / "cropwise_soil_tests_full.csv")
    soil["made_at"] = pd.to_datetime(soil["made_at"], errors="coerce")
    keep = [
        "field_id",
        "made_at",
        "elements.pH",
        "elements.organic_matter",
        "elements.N_NO3",
        "elements.P",
        "elements.K",
        "elements.Mg",
        "elements.S",
        "elements.Zn",
    ]
    soil = soil[[c for c in keep if c in soil.columns]].copy()
    return soil.rename(
        columns={
            "elements.pH": "field_soil_pH",
            "elements.organic_matter": "field_soil_OM",
            "elements.N_NO3": "field_soil_N",
            "elements.P": "field_soil_P",
            "elements.K": "field_soil_K",
            "elements.Mg": "field_soil_Mg",
            "elements.S": "field_soil_S",
            "elements.Zn": "field_soil_Zn",
        }
    )


def add_soil_asof(df: pd.DataFrame, soil: pd.DataFrame, month: int, day: int) -> pd.DataFrame:
    out = df.copy()
    soil_cols = [c for c in soil.columns if c not in {"field_id", "made_at"}]
    for c in soil_cols:
        out[c] = np.nan

    by_field = {int(fid): g.sort_values("made_at") for fid, g in soil.groupby("field_id")}
    for idx, row in out.iterrows():
        g = by_field.get(int(row["field_id"]))
        if g is None:
            continue
        asof = pd.Timestamp(year=int(row["year"]), month=month, day=day)
        past = g[g["made_at"].isna() | (g["made_at"] <= asof)]
        if past.empty:
            continue
        latest = past.iloc[-1]
        for c in soil_cols:
            out.at[idx, c] = latest.get(c)
    return out


def build_one(base: pd.DataFrame, sat: dict, soil: pd.DataFrame, tag: str, month: int, day: int) -> pd.DataFrame:
    rows = []
    for row in base.itertuples(index=False):
        rec = row._asdict()
        rec["asof_tag"] = tag
        rec.update(sowing_features(rec.get("sowing_date"), int(rec["year"]), month, day))
        for product, prefix in [("ndvi", "ndvi"), ("temperature", "cw_temp"), ("soil_moisture", "cw_sm")]:
            values = sat.get((int(rec["field_id"]), int(rec["year"]), product), [])
            rec.update(series_features(values, int(rec["year"]), month, day, prefix))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = add_soil_asof(out, soil, month, day)
    out = out[out["ndvi_n_obs_asof"] >= 3].reset_index(drop=True)
    return out


def main() -> None:
    base = load_base()
    print(f"Base target rows: {len(base):,}, fields={base['field_id'].nunique():,}, years={base['year'].min()}-{base['year'].max()}")
    sat = load_satellite(base)
    soil = load_soil_long()

    summary = []
    for tag, (month, day) in ASOF_DATES.items():
        out = build_one(base, sat, soil, tag, month, day)
        path = DATA_PROCESSED / f"ml_dataset_v23_cropwise_full_asof_{tag}.csv"
        out.to_csv(path, index=False)
        summary.append(
            {
                "asof_tag": tag,
                "rows": len(out),
                "fields": out["field_id"].nunique(),
                "years": f"{int(out['year'].min())}-{int(out['year'].max())}",
                "crops": out["standard_name"].nunique(),
                "soil_fields": out.loc[out["field_soil_pH"].notna(), "field_id"].nunique() if "field_soil_pH" in out else 0,
                "target_mean": out["target_yield_t_ha"].mean(),
                "top_crops": out["standard_name"].value_counts().head(8).to_dict(),
            }
        )
        print(f"{tag}: rows={len(out):,}, fields={out['field_id'].nunique():,}, cols={len(out.columns)} -> {path}")

    sm = pd.DataFrame(summary)
    md = ["# v23 Cropwise full dataset summary\n\n"]
    md.append("Source: refreshed Cropwise API pull (`fields`, `soil_tests`, `historical_values`).\n\n")
    md.append(sm.to_markdown(index=False))
    md.append("\n\n")
    md.append("Target policy: v17-compatible strict target; conflict rows dropped.\n")
    md.append("Satellite products used: `ndvi`, `temperature`, `soil_moisture`.\n")
    (REPORTS / "V23_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
