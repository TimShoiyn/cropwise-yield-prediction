"""
v9 = v6 clean feature set + factual targets from scratch + real sowing-date features.

Why:
  - v7 fixed target circularity by filtering away Cropwise estimates.
  - v9 additionally uses real sowing_date from history_items_full.csv instead of
    calendar approximations (April 1) for GDD/NDVI/stress alignment.

Outputs:
  data_processed/ml_dataset_clean_v9_asof_{tag}.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")

ASOF_TAGS = ["07_01", "08_01", "09_01"]
GDD_STAGES = [200, 400, 600, 800, 1000, 1200]
DAS_STAGES = [30, 45, 60, 75, 90, 120]
VPD_HIGH_KPA = 2.0
HOT_DAY_C = 30.0
DRY_DAY_MM = 1.0

CROP_BASE_TEMP = {
    "sunflower": 6.0,
    "wheat_spring": 4.0,
    "wheat_winter": 4.0,
    "barley_spring": 4.0,
    "barley_winter": 4.0,
    "oil_seed_raps_spring": 5.0,
    "oil_seed_raps_winter": 5.0,
    "maize": 8.0,
    "soya": 8.0,
    "pea": 4.0,
}
DEFAULT_BASE_TEMP = 5.0


def valid_sowing_date(std_name: str | None, sowing_date: pd.Timestamp, as_of: pd.Timestamp) -> bool:
    if pd.isna(sowing_date) or sowing_date >= as_of:
        return False
    das = (as_of - sowing_date).days
    if das < 0 or das > 430:
        return False
    month = sowing_date.month
    if std_name == "wheat_winter":
        return month in {8, 9, 10}
    if std_name in {"sunflower", "wheat_spring", "barley_spring", "oil_seed_raps_spring", "pea", "soya"}:
        return month in {4, 5, 6}
    return True


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for item in mask:
        if bool(item):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def load_ndvi() -> pd.DataFrame:
    ndvi = pd.read_csv(DATA_RAW / "ndvi_timeseries.csv", usecols=["field_id", "date", "ndvi_mean", "cloud_coverage"])
    ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
    ndvi = ndvi.dropna(subset=["date", "ndvi_mean"]).copy()
    ndvi = ndvi[ndvi["cloud_coverage"].fillna(0) <= 30]
    ndvi["field_id"] = ndvi["field_id"].astype(int)
    daily = (
        ndvi.groupby(["field_id", "date"])["ndvi_mean"]
        .mean()
        .reset_index()
        .sort_values(["field_id", "date"])
    )
    parts = []
    for fid, g in daily.groupby("field_id"):
        g = g.sort_values("date").copy()
        g["ndvi_smooth"] = g["ndvi_mean"].rolling(7, center=True, min_periods=2).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def interp_at_x(g: pd.DataFrame, x_col: str, y_col: str, x: float) -> float:
    if g.empty or x < g[x_col].min() or x > g[x_col].max():
        return float("nan")
    xp = g[x_col].to_numpy(dtype=float)
    fp = g[y_col].to_numpy(dtype=float)
    mask = np.isfinite(xp) & np.isfinite(fp)
    if mask.sum() < 2:
        return float("nan")
    order = np.argsort(xp[mask])
    return float(np.interp(x, xp[mask][order], fp[mask][order]))


def compute_one_row(row: pd.Series, weather: pd.DataFrame, ndvi: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    fid = int(row["field_id"])
    std_name = row.get("standard_name")
    sowing_date = pd.to_datetime(row.get("sowing_date"), errors="coerce")
    out: dict[str, float | int] = {
        "field_id": fid,
        "year": int(row["year"]),
        "sowing_date_valid": 0,
    }
    if not valid_sowing_date(std_name, sowing_date, as_of):
        return out

    out["sowing_date_valid"] = 1
    out["days_after_sowing_asof"] = int((as_of - sowing_date).days)
    out["sowing_doy"] = int(sowing_date.dayofyear)

    w = weather[
        (weather["field_id"] == fid)
        & (weather["date"] >= sowing_date)
        & (weather["date"] < as_of)
    ].sort_values("date")
    base_temp = CROP_BASE_TEMP.get(std_name or "", DEFAULT_BASE_TEMP)
    out["gdd_base_temp"] = float(base_temp)
    if not w.empty:
        temp = pd.to_numeric(w["temperature_2m_mean"], errors="coerce")
        tmax = pd.to_numeric(w["temperature_2m_max"], errors="coerce")
        precip = pd.to_numeric(w["precipitation_sum"], errors="coerce")
        et0 = pd.to_numeric(w["et0_fao_evapotranspiration"], errors="coerce")
        vpd = pd.to_numeric(w["vapour_pressure_deficit_max"], errors="coerce")
        srad = pd.to_numeric(w["shortwave_radiation_sum"], errors="coerce")
        gdd_daily = np.clip(temp.fillna(0).to_numpy(dtype=float) - base_temp, 0, None)
        w = w.copy()
        w["gdd_from_sowing"] = np.cumsum(gdd_daily)
        out["gdd_from_sowing_asof"] = float(w["gdd_from_sowing"].max())
        out["wx_sow_precip_sum"] = float(np.nansum(precip))
        out["wx_sow_et0_sum"] = float(np.nansum(et0))
        out["wx_sow_water_balance"] = out["wx_sow_precip_sum"] - out["wx_sow_et0_sum"]
        out["wx_sow_srad_sum"] = float(np.nansum(srad))
        out["wx_sow_vpd_max"] = float(np.nanmax(vpd)) if np.isfinite(vpd).any() else float("nan")
        out["wx_sow_vpd_days_high"] = int((vpd >= VPD_HIGH_KPA).sum())
        out["wx_sow_hot_d30"] = int((tmax >= HOT_DAY_C).sum())
        out["wx_sow_dry_days"] = int((precip < DRY_DAY_MM).sum())
        out["wx_sow_drought_run"] = longest_run((precip < DRY_DAY_MM).fillna(False).to_numpy())
    else:
        w = pd.DataFrame()

    n = ndvi[
        (ndvi["field_id"] == fid)
        & (ndvi["date"] >= sowing_date)
        & (ndvi["date"] < as_of)
    ].sort_values("date")
    if not n.empty:
        n = n.copy()
        n["days_after_sowing"] = (n["date"] - sowing_date).dt.days
        out["ndvi_sow_n_obs"] = int(n["ndvi_smooth"].notna().sum())
        out["ndvi_sow_mean"] = float(np.nanmean(n["ndvi_smooth"]))
        out["ndvi_sow_max"] = float(np.nanmax(n["ndvi_smooth"]))
        out["ndvi_sow_last"] = float(n.dropna(subset=["ndvi_smooth"])["ndvi_smooth"].iloc[-1])
        out["ndvi_sow_peak_das"] = int(n.loc[n["ndvi_smooth"].idxmax(), "days_after_sowing"])
        for das in DAS_STAGES:
            out[f"ndvi_at_das_{das}"] = interp_at_x(n, "days_after_sowing", "ndvi_smooth", float(das))

        if not w.empty:
            n_gdd = n.merge(w[["date", "gdd_from_sowing"]], on="date", how="inner")
            if not n_gdd.empty:
                out["ndvi_sow_peak_gdd"] = float(n_gdd.loc[n_gdd["ndvi_smooth"].idxmax(), "gdd_from_sowing"])
                for stage in GDD_STAGES:
                    out[f"ndvi_at_sow_gdd_{stage}"] = interp_at_x(
                        n_gdd, "gdd_from_sowing", "ndvi_smooth", float(stage)
                    )
                early = n_gdd[(n_gdd["ndvi_smooth"] >= 0.20) & n_gdd["ndvi_smooth"].notna()]
                if not early.empty:
                    first = early.iloc[0]
                    peak = n_gdd.loc[n_gdd["ndvi_smooth"].idxmax()]
                    dg = float(peak["gdd_from_sowing"] - first["gdd_from_sowing"])
                    if dg > 50:
                        out["ndvi_growth_per_100gdd_from_sow"] = float(
                            (peak["ndvi_smooth"] - first["ndvi_smooth"]) / dg * 100.0
                        )
    return out


def build_for_tag(tag: str, weather: pd.DataFrame, ndvi: pd.DataFrame, targets: pd.DataFrame) -> None:
    base_path = DATA_PROCESSED / f"ml_dataset_clean_v6_asof_{tag}.csv"
    out_path = DATA_PROCESSED / f"ml_dataset_clean_v9_asof_{tag}.csv"
    base = pd.read_csv(base_path)

    # Remove target columns from v6, then attach factual target from scratch.
    drop_target_cols = ["target_yield_t_ha", "target_source", "unit_fix_applied"]
    base = base.drop(columns=[c for c in drop_target_cols if c in base.columns])
    factual = targets[targets["kept"]].copy()
    factual_cols = [
        "field_id",
        "year",
        "target_yield_t_ha",
        "target_source",
        "unit_fix_applied",
        "sowing_date",
        "harvesting_date",
    ]
    df = base.merge(factual[factual_cols], on=["field_id", "year"], how="inner")
    df["sowing_date"] = pd.to_datetime(df["sowing_date"], errors="coerce")

    rows = []
    mm, dd = map(int, tag.split("_"))
    for _, row in df.iterrows():
        as_of = pd.Timestamp(year=int(row["year"]), month=mm, day=dd)
        rows.append(compute_one_row(row, weather, ndvi, as_of))
    sow_feats = pd.DataFrame(rows)
    out = df.merge(sow_feats, on=["field_id", "year"], how="left")
    out.to_csv(out_path, index=False)

    new_cols = [c for c in sow_feats.columns if c not in {"field_id", "year"}]
    coverage = out[new_cols].notna().mean().sort_values()
    print(f"  {tag}: {len(base)} base rows -> {len(out)} factual rows, {out.shape[1]} cols")
    print(f"    sowing_date_valid: {out['sowing_date_valid'].mean() * 100:.1f}%")
    print(f"    low coverage (<50%): {coverage[coverage < 0.5].to_dict()}")


def main() -> None:
    print("=" * 80)
    print("BUILD v9 — FACTUAL TARGET + REAL SOWING FEATURES")
    print("=" * 80)

    targets = pd.read_csv(DATA_PROCESSED / "targets_factual_t_ha.csv")
    weather = pd.read_csv(
        DATA_RAW / "openmeteo_daily.csv",
        usecols=[
            "field_id",
            "date",
            "temperature_2m_mean",
            "temperature_2m_max",
            "precipitation_sum",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit_max",
        ],
    )
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["field_id"] = weather["field_id"].astype(int)

    ndvi = load_ndvi()
    print(f"Weather rows: {len(weather)}")
    print(f"NDVI rows: {len(ndvi)}")
    for tag in ASOF_TAGS:
        build_for_tag(tag, weather, ndvi, targets)
    print("\nDone.")


if __name__ == "__main__":
    main()
