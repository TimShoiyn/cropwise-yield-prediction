"""
Sprint 5.1 — as-of feature dataset v3 backed by Open-Meteo.

What changes vs v2b:
  - Drop ALL old wx_* features (built from weather_history_items.csv
    which was sparse for 2010-2014 and lacked solar radiation/VPD/ET0).
  - Rebuild weather features from data_raw/openmeteo_daily.csv (ERA5).
  - Add brand-new agronomic signals:
      * wx_srad_sum_to_asof / monthly  (light = photosynthesis ceiling)
      * wx_vpd_max_to_asof, wx_vpd_days_high_to_asof  (atmospheric drought)
      * wx_et0_sum_to_asof + wx_water_balance_to_asof  (water demand vs supply)
      * wx_gdd_sunflower_to_asof (base 6°C, from 2-week sowing window)
      * wx_gdd_wheat_to_asof     (base 4°C)
      * monthly aggregates for fully-elapsed months (apr/may/jun/jul/aug)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")

OM_PATH = DATA_RAW / "openmeteo_daily.csv"
ASOF_DATES = ["07_01", "08_01", "09_01"]

GDD_BASE_SUNFLOWER = 6.0
GDD_BASE_WHEAT = 4.0
VPD_HIGH_THRESHOLD_KPA = 2.0
HOT_DAY_T30 = 30.0
HOT_DAY_T35 = 35.0
DRY_DAY_PRECIP_MM = 1.0


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def aggregate_until(om: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Aggregate Open-Meteo daily rows up to (but not including) as_of."""
    df = om[om["date"] < as_of].copy()
    if df.empty:
        return pd.DataFrame()

    year = as_of.year
    apr1 = pd.Timestamp(year=year, month=4, day=1)
    season = df[df["date"] >= apr1].copy()
    last30 = df[df["date"] >= as_of - pd.Timedelta(days=30)].copy()

    out_rows = []
    for fid, g in season.groupby("field_id"):
        g = g.sort_values("date")
        t_mean = g["temperature_2m_mean"].to_numpy()
        t_max = g["temperature_2m_max"].to_numpy()
        precip = g["precipitation_sum"].to_numpy()
        srad = g["shortwave_radiation_sum"].to_numpy()
        vpd = g["vapour_pressure_deficit_max"].to_numpy()
        et0 = g["et0_fao_evapotranspiration"].to_numpy()

        gdd_sf = np.clip(t_mean - GDD_BASE_SUNFLOWER, 0, None).sum()
        gdd_wh = np.clip(t_mean - GDD_BASE_WHEAT, 0, None).sum()

        hot_d30 = int((t_max >= HOT_DAY_T30).sum())
        hot_d35 = int((t_max >= HOT_DAY_T35).sum())
        dry_mask = precip < DRY_DAY_PRECIP_MM
        dry_days = int(dry_mask.sum())
        drought_run = _longest_run(dry_mask)
        heat_mask = t_max >= HOT_DAY_T30
        heat_run = _longest_run(heat_mask)
        vpd_high_days = int((vpd >= VPD_HIGH_THRESHOLD_KPA).sum())
        precip_sum = float(np.nansum(precip))
        et0_sum = float(np.nansum(et0))
        water_balance = precip_sum - et0_sum

        last30_g = last30[last30["field_id"] == fid]
        srad_30 = float(np.nansum(last30_g["shortwave_radiation_sum"])) if not last30_g.empty else np.nan
        et0_30 = float(np.nansum(last30_g["et0_fao_evapotranspiration"])) if not last30_g.empty else np.nan
        precip_30 = float(np.nansum(last30_g["precipitation_sum"])) if not last30_g.empty else np.nan
        vpd_max_30 = float(np.nanmax(last30_g["vapour_pressure_deficit_max"])) if not last30_g.empty else np.nan
        t_max_30 = float(np.nanmax(last30_g["temperature_2m_max"])) if not last30_g.empty else np.nan

        row = {
            "field_id": int(fid),
            "year": year,
            "wx_temp_mean_to_asof": float(np.nanmean(t_mean)),
            "wx_temp_max_to_asof": float(np.nanmax(t_max)),
            "wx_precip_sum_to_asof": precip_sum,
            "wx_srad_sum_to_asof": float(np.nansum(srad)),
            "wx_srad_mean_to_asof": float(np.nanmean(srad)),
            "wx_vpd_max_to_asof": float(np.nanmax(vpd)),
            "wx_vpd_mean_to_asof": float(np.nanmean(vpd)),
            "wx_vpd_days_high_to_asof": vpd_high_days,
            "wx_et0_sum_to_asof": et0_sum,
            "wx_water_balance_to_asof": water_balance,
            "wx_gdd_sunflower_to_asof": float(gdd_sf),
            "wx_gdd_wheat_to_asof": float(gdd_wh),
            "wx_hot_d30_to_asof": hot_d30,
            "wx_hot_d35_to_asof": hot_d35,
            "wx_dry_days_to_asof": dry_days,
            "wx_drought_run_to_asof": drought_run,
            "wx_heat_run_to_asof": heat_run,
            "wx_precip_sum_last30d": precip_30,
            "wx_srad_sum_last30d": srad_30,
            "wx_et0_sum_last30d": et0_30,
            "wx_vpd_max_last30d": vpd_max_30,
            "wx_temp_max_last30d": t_max_30,
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    # Per-month aggregates for fully elapsed months (start_month <= as_of_month - 1)
    months_elapsed = list(range(4, as_of.month))
    if months_elapsed:
        for m in months_elapsed:
            month_df = df[df["date"].dt.month == m]
            if month_df.empty:
                continue
            agg = (
                month_df.groupby("field_id")
                .agg(
                    precip=("precipitation_sum", "sum"),
                    temp_mean=("temperature_2m_mean", "mean"),
                    temp_max=("temperature_2m_max", "max"),
                    srad=("shortwave_radiation_sum", "sum"),
                    et0=("et0_fao_evapotranspiration", "sum"),
                    vpd_max=("vapour_pressure_deficit_max", "max"),
                    hot_d30=("temperature_2m_max", lambda s: int((s >= HOT_DAY_T30).sum())),
                )
                .reset_index()
            )
            month_name = ["", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][m]
            agg = agg.rename(
                columns={
                    "precip": f"wx_{month_name}_precip",
                    "temp_mean": f"wx_{month_name}_temp_mean",
                    "temp_max": f"wx_{month_name}_temp_max",
                    "srad": f"wx_{month_name}_srad",
                    "et0": f"wx_{month_name}_et0",
                    "vpd_max": f"wx_{month_name}_vpd_max",
                    "hot_d30": f"wx_{month_name}_hot_d30",
                }
            )
            out = out.merge(agg, on="field_id", how="left")

    return out


def build_for_asof(asof_tag: str) -> None:
    in_path = DATA_PROCESSED / f"ml_dataset_clean_v2b_asof_{asof_tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v3_asof_{asof_tag}.csv"
    print(f"\nas-of {asof_tag}")
    print(f"  input:  {in_path.name}")

    base = pd.read_csv(in_path)
    # Drop old wx_* columns — they will be replaced
    drop_cols = [c for c in base.columns if c.startswith("wx_")]
    print(f"  dropping {len(drop_cols)} old wx_* columns")
    base = base.drop(columns=drop_cols)

    om = pd.read_csv(OM_PATH)
    om["date"] = pd.to_datetime(om["date"], errors="coerce")

    parts: list[pd.DataFrame] = []
    for year, g in base.groupby("year"):
        mm, dd = asof_tag.split("_")
        as_of = pd.Timestamp(year=int(year), month=int(mm), day=int(dd))
        wx_year = aggregate_until(om[om["year"] == year], as_of)
        if wx_year.empty:
            parts.append(g)
            continue
        merged = g.merge(wx_year, on=["field_id", "year"], how="left")
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"  output: {out_path.name} → {len(out)} rows × {out.shape[1]} cols")


def main():
    print("=" * 80)
    print("BUILD AS-OF FEATURES v3 (Open-Meteo backed)")
    print("=" * 80)
    if not OM_PATH.exists():
        raise FileNotFoundError(f"Missing {OM_PATH}. Run scripts/external/fetch_open_meteo.py first.")
    for tag in ASOF_DATES:
        build_for_asof(tag)
    print("\nDone.")


if __name__ == "__main__":
    main()
