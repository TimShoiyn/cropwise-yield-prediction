"""
Build v18 datasets from data_raw/productivity_estimate_peers.csv.

The peers file is much larger than our 30-field dataset and already contains
per-row NDVI time series. We build three as-of datasets (1 Jul, 1 Aug, 1 Sep).

Two variants are emitted:
  - strict: only clearly as-of-safe data:
      crop, area, variety, previous crop, sowing date, NDVI values truncated
      to the as-of date.
  - plus: strict + peers' aggregate weather/soil proxy columns
      (accumulated_precipitations, average_soil_moisture). These are useful
      but treated as exploratory because the source does not document their
      exact as-of horizon.

Target:
  productivity is stored in c/ha, converted to t/ha by /10.

Outputs:
  data_processed/ml_dataset_v18_peers_{strict,plus}_asof_{07_01,08_01,09_01}.csv
  reports/V18_PEERS_DATASET_SUMMARY.md
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")

ASOF_DATES = {
    "07_01": (7, 1),
    "08_01": (8, 1),
    "09_01": (9, 1),
}

TARGET_CAPS_T_HA = {
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
    "linum": 3.0,
    "rye_winter": 6.0,
    "safflower": 2.5,
    "potatoes": 60.0,
    "sugar_beet": 80.0,
}

DEDUP_COLS = [
    "year",
    "productivity",
    "area",
    "crop",
    "variety",
    "previous_crop_name",
    "previous_crop_productivity_estimate_crop_name",
    "previous_crop_standard_name",
    "accumulated_precipitations",
    "average_soil_moisture",
    "sowing_date",
    "harvesting_date",
    "ndvi_values",
]


def parse_ndvi_values(value: object) -> list[tuple[pd.Timestamp, float]]:
    if pd.isna(value):
        return []
    try:
        raw = ast.literal_eval(str(value))
    except Exception:
        return []
    parsed: list[tuple[pd.Timestamp, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        date = pd.to_datetime(item[0], errors="coerce")
        ndvi = pd.to_numeric(item[1], errors="coerce")
        if pd.notna(date) and pd.notna(ndvi):
            parsed.append((pd.Timestamp(date), float(ndvi)))
    parsed.sort(key=lambda x: x[0])
    return parsed


def ndvi_features(values: list[tuple[pd.Timestamp, float]], year: int, month: int, day: int) -> dict[str, float]:
    asof = pd.Timestamp(year=int(year), month=month, day=day)
    vals = [(d, v) for d, v in values if d <= asof and d.year == int(year)]
    if not vals:
        return {
            "ndvi_n_obs_asof": 0,
            "ndvi_mean_asof": np.nan,
            "ndvi_max_asof": np.nan,
            "ndvi_min_asof": np.nan,
            "ndvi_std_asof": np.nan,
            "ndvi_p25_asof": np.nan,
            "ndvi_p75_asof": np.nan,
            "ndvi_last_value_asof": np.nan,
            "ndvi_last_doy_asof": np.nan,
            "ndvi_peak_doy_asof": np.nan,
            "ndvi_amplitude_asof": np.nan,
            "ndvi_slope_asof": np.nan,
            "ndvi_integral_asof": np.nan,
            "ndvi_last30d_mean_asof": np.nan,
            "ndvi_above_05_n_asof": 0,
        }

    dates = np.array([d for d, _ in vals])
    ndvi = np.array([v for _, v in vals], dtype=float)
    doy = np.array([pd.Timestamp(d).dayofyear for d in dates], dtype=float)
    peak_idx = int(np.nanargmax(ndvi))
    slope = np.nan
    if len(ndvi) >= 2 and np.nanstd(doy) > 0:
        slope = float(np.polyfit(doy, ndvi, deg=1)[0])

    last30_start = asof - pd.Timedelta(days=30)
    last30 = np.array([v for d, v in vals if d >= last30_start], dtype=float)

    return {
        "ndvi_n_obs_asof": int(len(ndvi)),
        "ndvi_mean_asof": float(np.nanmean(ndvi)),
        "ndvi_max_asof": float(np.nanmax(ndvi)),
        "ndvi_min_asof": float(np.nanmin(ndvi)),
        "ndvi_std_asof": float(np.nanstd(ndvi)),
        "ndvi_p25_asof": float(np.nanpercentile(ndvi, 25)),
        "ndvi_p75_asof": float(np.nanpercentile(ndvi, 75)),
        "ndvi_last_value_asof": float(ndvi[-1]),
        "ndvi_last_doy_asof": float(doy[-1]),
        "ndvi_peak_doy_asof": float(doy[peak_idx]),
        "ndvi_amplitude_asof": float(np.nanmax(ndvi) - np.nanmin(ndvi)),
        "ndvi_slope_asof": slope,
        "ndvi_integral_asof": float(np.trapezoid(ndvi, doy)) if len(ndvi) >= 2 else 0.0,
        "ndvi_last30d_mean_asof": float(np.nanmean(last30)) if len(last30) else np.nan,
        "ndvi_above_05_n_asof": int(np.sum(ndvi >= 0.5)),
    }


def sowing_features(sowing_date: object, year: int, month: int, day: int) -> dict[str, float]:
    asof = pd.Timestamp(year=int(year), month=month, day=day)
    sow = pd.to_datetime(sowing_date, errors="coerce")
    if pd.isna(sow) or sow > asof:
        return {
            "sowing_known_asof": 0,
            "sowing_doy": np.nan,
            "days_after_sowing_asof": np.nan,
        }
    return {
        "sowing_known_asof": 1,
        "sowing_doy": float(pd.Timestamp(sow).dayofyear),
        "days_after_sowing_asof": float((asof - pd.Timestamp(sow)).days),
    }


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "crop",
        "variety",
        "previous_crop_name",
        "previous_crop_productivity_estimate_crop_name",
        "previous_crop_standard_name",
    ]:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def load_and_clean_peers() -> pd.DataFrame:
    peers = pd.read_csv(DATA_RAW / "productivity_estimate_peers.csv", low_memory=False)
    peers = peers.dropna(subset=["year", "productivity", "crop", "ndvi_values"]).copy()
    peers["year"] = pd.to_numeric(peers["year"], errors="coerce").astype("Int64")
    peers["target_yield_t_ha"] = pd.to_numeric(peers["productivity"], errors="coerce") / 10.0
    peers["area"] = pd.to_numeric(peers["area"], errors="coerce")
    peers["accumulated_precipitations"] = pd.to_numeric(peers["accumulated_precipitations"], errors="coerce")
    peers["average_soil_moisture"] = pd.to_numeric(peers["average_soil_moisture"], errors="coerce")
    peers = peers.dropna(subset=["year", "target_yield_t_ha"]).copy()
    peers["year"] = peers["year"].astype(int)
    peers = normalize_text_columns(peers)

    # Keep only plausible target values per crop.
    cap = peers["crop"].map(TARGET_CAPS_T_HA).fillna(12.0)
    peers = peers[(peers["target_yield_t_ha"] > 0.05) & (peers["target_yield_t_ha"] <= cap)].copy()

    before = len(peers)
    peers = peers.drop_duplicates(subset=DEDUP_COLS).reset_index(drop=True)
    peers["peer_row_id"] = np.arange(len(peers))
    print(f"Loaded peers rows: {before:,}; after exact dedup: {len(peers):,}")
    print("Parsing NDVI lists once...")
    peers["parsed_ndvi_values"] = peers["ndvi_values"].map(parse_ndvi_values)
    return peers


def build_one(peers: pd.DataFrame, tag: str, month: int, day: int, variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, row in peers.iterrows():
        features: dict[str, object] = {
            "peer_row_id": row["peer_row_id"],
            "year": int(row["year"]),
            "standard_name": row["crop"],
            "crop": row["crop"],
            "variety": row["variety"],
            "previous_crop_name": row["previous_crop_name"],
            "previous_crop_productivity_estimate_crop_name": row["previous_crop_productivity_estimate_crop_name"],
            "previous_crop_standard_name": row["previous_crop_standard_name"],
            "area": row["area"],
            "target_yield_t_ha": row["target_yield_t_ha"],
            "target_source": "productivity_estimate_peers_productivity_c_ha",
            "asof_tag": tag,
        }
        features.update(ndvi_features(row["parsed_ndvi_values"], int(row["year"]), month, day))
        features.update(sowing_features(row["sowing_date"], int(row["year"]), month, day))
        if variant == "plus":
            features["accumulated_precipitations"] = row["accumulated_precipitations"]
            features["average_soil_moisture"] = row["average_soil_moisture"]
        rows.append(features)
    out = pd.DataFrame(rows)
    out = out[out["ndvi_n_obs_asof"] >= 3].reset_index(drop=True)
    return out


def summarize(df: pd.DataFrame, tag: str, variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "asof_tag": tag,
        "rows": len(df),
        "years": f"{int(df['year'].min())}-{int(df['year'].max())}",
        "crops": df["standard_name"].nunique(),
        "target_mean": df["target_yield_t_ha"].mean(),
        "target_min": df["target_yield_t_ha"].min(),
        "target_max": df["target_yield_t_ha"].max(),
        "top_crops": df["standard_name"].value_counts().head(8).to_dict(),
    }


def main() -> None:
    peers = load_and_clean_peers()
    summaries: list[dict[str, object]] = []
    for tag, (month, day) in ASOF_DATES.items():
        for variant in ["strict", "plus"]:
            out = build_one(peers, tag, month, day, variant)
            path = DATA_PROCESSED / f"ml_dataset_v18_peers_{variant}_asof_{tag}.csv"
            out.to_csv(path, index=False)
            summaries.append(summarize(out, tag, variant))
            print(f"Wrote {path}: rows={len(out):,}, cols={len(out.columns)}")

    summary = pd.DataFrame(summaries)
    md = ["# v18 peers dataset summary\n\n"]
    md.append("Source: `data_raw/productivity_estimate_peers.csv`.\n\n")
    md.append("- Target: `productivity / 10` => t/ha.\n")
    md.append("- `strict`: crop/area/previous crop/sowing + NDVI truncated to as-of.\n")
    md.append("- `plus`: strict + `accumulated_precipitations`, `average_soil_moisture` (exploratory, horizon not documented).\n")
    md.append("- Raw `max_ndvi` is intentionally excluded because it can be full-season leakage.\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V18_PEERS_DATASET_SUMMARY.md").write_text("".join(md), encoding="utf-8")
    print(f"Wrote {REPORTS / 'V18_PEERS_DATASET_SUMMARY.md'}")


if __name__ == "__main__":
    main()
