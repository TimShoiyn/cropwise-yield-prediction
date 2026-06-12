"""
v22: local compact-core model + leakage-safe historical field baseline.

Why:
  Audit-3b showed that 148 sparse features overfit on 180 rows.
  The core 24-feature set improved local walk-forward R2 and MAPE.

This script turns that finding into a clean experiment across all as-of dates:
  - core: compact NDVI/weather/sowing/basic soil features
  - core_history: core + past field yield baseline features computed only from
    years < test_year
  - Cropwise benchmark on identical rows

Validation:
  Walk-forward by year. For each test year, model sees only previous years.

Outputs:
  models_v2/asof_comparison/asof_results_v22_core_history.csv
  models_v2/asof_comparison/asof_predictions_v22_core_history.csv
  reports/V22_CORE_HISTORY_FINAL_RU.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
COMP_DIR.mkdir(parents=True, exist_ok=True)

ASOF_TAGS = ["07_01", "08_01", "09_01"]
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
SCENARIOS = [
    ("all_crops", None),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("sunflower", ["sunflower"]),
]

CORE_FEATURES = [
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_p75_asof",
    "ndvi_integral_asof",
    "ndvi_amplitude_asof",
    "ndvi_anom_mean_asof",
    "ndvi_above_05_days_asof",
    "ndvi_slope_asof",
    "ndvi_std_asof",
    "wx_precip_sum_to_asof",
    "wx_temp_mean_to_asof",
    "wx_vpd_max_to_asof",
    "wx_water_balance_to_asof",
    "wx_srad_sum_to_asof",
    "wx_may_precip",
    "wx_jul_vpd_max",
    "wx_may_srad",
    "days_after_sowing_asof",
    "sowing_doy",
    "years_since_wheat",
    "years_since_sunflower",
    "field_tillable_area",
    "field_soil_pH",
    "field_soil_OM",
]

HISTORY_FEATURES = [
    "hist_field_mean_yield",
    "hist_field_crop_mean_yield",
    "hist_field_dev_mean",
    "hist_field_n",
    "hist_field_crop_n",
]

PARAMS = dict(
    iterations=500,
    learning_rate=0.045,
    depth=4,
    l2_leaf_reg=30.0,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) < 3:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100.0),
    }


def within_crop_year_r2(df: pd.DataFrame, pred_col: str) -> float:
    sub = df.dropna(subset=["target_yield_t_ha", pred_col, "standard_name", "year"]).copy()
    if len(sub) < 4:
        return np.nan
    y = sub["target_yield_t_ha"].astype(float)
    p = sub[pred_col].astype(float)
    g = [sub["standard_name"], sub["year"]]
    y_dm = y - y.groupby(g).transform("mean")
    p_dm = p - p.groupby(g).transform("mean")
    ss_tot = float((y_dm**2).sum())
    if ss_tot <= 1e-9:
        return np.nan
    return 1.0 - float(((y_dm - p_dm) ** 2).sum()) / ss_tot


def load_local(tag: str, crop_filter: list[str] | None, cw: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v17_v12_clean_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = attach_cropwise(df, tag, cw)
    return df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)


def add_history_features_for_year(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Compute past-yield features using train years only."""
    out = test.copy()
    global_mean = float(train["target_yield_t_ha"].mean())

    # Past field/crop means.
    field_mean = train.groupby("field_id")["target_yield_t_ha"].mean()
    field_n = train.groupby("field_id")["target_yield_t_ha"].count()
    field_crop_mean = train.groupby(["field_id", "standard_name"])["target_yield_t_ha"].mean()
    field_crop_n = train.groupby(["field_id", "standard_name"])["target_yield_t_ha"].count()

    # Past deviation from crop-year average.
    t = train.copy()
    t["crop_year_mean"] = t.groupby(["standard_name", "year"])["target_yield_t_ha"].transform("mean")
    t["field_dev"] = t["target_yield_t_ha"] - t["crop_year_mean"]
    field_dev = t.groupby("field_id")["field_dev"].mean()

    out["hist_field_mean_yield"] = out["field_id"].map(field_mean).fillna(global_mean)
    out["hist_field_n"] = out["field_id"].map(field_n).fillna(0)
    out["hist_field_dev_mean"] = out["field_id"].map(field_dev).fillna(0.0)

    fc_index = pd.MultiIndex.from_frame(out[["field_id", "standard_name"]])
    out["hist_field_crop_mean_yield"] = pd.Series(fc_index.map(field_crop_mean), index=out.index).fillna(
        out["hist_field_mean_yield"]
    )
    out["hist_field_crop_n"] = pd.Series(fc_index.map(field_crop_n), index=out.index).fillna(0)
    return out


def prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    use = [c for c in features if c in df.columns]
    return df[use].apply(pd.to_numeric, errors="coerce")


def walk_forward_predict(df: pd.DataFrame, features: list[str], use_history: bool) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for test_year in sorted(df["year"].unique()):
        train = df[df["year"] < test_year].copy()
        test_mask = df["year"].values == test_year
        test = df[test_mask].copy()
        if len(train) < 20:
            continue
        feat = list(features)
        if use_history:
            train = add_history_features_for_year(train, train)
            test = add_history_features_for_year(df[df["year"] < test_year].copy(), test)
            feat = feat + HISTORY_FEATURES
        model = CatBoostRegressor(**PARAMS)
        model.fit(Pool(prepare_X(train, feat), train["target_yield_t_ha"].astype(float)))
        pred[test_mask] = model.predict(prepare_X(test, feat))
    return pred


def evaluate_block(df: pd.DataFrame, scenario: str, tag: str, name: str, pred_col: str) -> dict:
    cmp_df = df.dropna(subset=[pred_col]).copy()
    m = metrics(cmp_df["target_yield_t_ha"], cmp_df[pred_col])
    return {
        "scenario": scenario,
        "asof_tag": tag,
        "asof": ASOF_LABELS[tag],
        "model": name,
        "n": len(cmp_df),
        "r2": m["r2"],
        "wcy_r2": within_crop_year_r2(cmp_df, pred_col),
        "rmse": m["rmse"],
        "mae": m["mae"],
        "mape": m["mape"],
    }


def main() -> None:
    print("=" * 80)
    print("v22 CORE LOCAL + HISTORY BASELINE")
    print("=" * 80)
    cw = cropwise_table()
    rows = []
    pred_rows = []

    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            df = load_local(tag, crop_filter, cw)
            df["pred_core"] = walk_forward_predict(df, CORE_FEATURES, use_history=False)
            df["pred_core_history"] = walk_forward_predict(df, CORE_FEATURES, use_history=True)
            for col, name in [
                ("pred_core", "v22_core"),
                ("pred_core_history", "v22_core_history"),
                ("cropwise_asof_t_ha", "cropwise"),
            ]:
                if col == "cropwise_asof_t_ha":
                    block_df = df.dropna(subset=[col])
                else:
                    block_df = df
                row = evaluate_block(block_df, scenario, tag, name, col)
                rows.append(row)
            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_core",
                    "pred_core_history",
                ]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)
            r_core = rows[-3]
            r_hist = rows[-2]
            r_cw = rows[-1]
            print(
                f"  {tag}: core MAPE={r_core['mape']:.1f}% R2={r_core['r2']:.3f} wcy={r_core['wcy_r2']:.3f}; "
                f"hist MAPE={r_hist['mape']:.1f}% R2={r_hist['r2']:.3f} wcy={r_hist['wcy_r2']:.3f}; "
                f"CW MAPE={r_cw['mape']:.1f}% R2={r_cw['r2']:.3f} wcy={r_cw['wcy_r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v22_core_history.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v22_core_history.csv", index=False)

    md = ["# v22: core local model + historical field baseline\n\n"]
    md.append(
        "Validation: walk-forward by year. `v22_core` uses 24 compact features; "
        "`v22_core_history` adds only past-year field yield baseline features. "
        "`wcy_r2` = within-crop-year R2.\n\n"
    )
    md.append(res.round(3).to_markdown(index=False) + "\n\n")

    # Practical winners by MAPE and wcy_r2.
    md.append("## Winners by scenario/date\n\n")
    for scenario, _ in SCENARIOS:
        for tag in ASOF_TAGS:
            sub = res[(res["scenario"] == scenario) & (res["asof_tag"] == tag)].copy()
            best_mape = sub.sort_values("mape").iloc[0]
            best_wcy = sub.sort_values("wcy_r2", ascending=False).iloc[0]
            md.append(
                f"- {scenario} {ASOF_LABELS[tag]}: best MAPE = **{best_mape['model']}** "
                f"({best_mape['mape']:.1f}%), best wcy_R2 = **{best_wcy['model']}** "
                f"({best_wcy['wcy_r2']:.3f}).\n"
            )
    md.append("\n")
    (REPORTS / "V22_CORE_HISTORY_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print(f"\nSaved {COMP_DIR / 'asof_results_v22_core_history.csv'}")
    print(f"Saved {REPORTS / 'V22_CORE_HISTORY_FINAL_RU.md'}")


if __name__ == "__main__":
    main()
