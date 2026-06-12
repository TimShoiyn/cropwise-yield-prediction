"""
v20 domain-adapted residual calibration.

v18 gave a global peer model:
    peer_pred = f_peer(NDVI, crop, sowing, previous crop, area)

v19 calibrated only peer_pred + crop. v20 learns a local residual model:
    residual = target_yield_t_ha - peer_pred
    residual = f_local(local NDVI anomalies, local weather, sowing, soil/management, crop)

Validation remains leakage-safe:
  - for each local test year Y, residual calibration uses only local rows
    with year < Y;
  - Cropwise is never used as a feature;
  - target-source / post-harvest columns are excluded.

Inputs:
  - models_v2/asof_comparison/asof_predictions_v18_peer_transfer.csv
  - data_processed/ml_dataset_clean_v17_v15_rates_clean_asof_*.csv
    (falls back to v17_v12_clean if rates file is missing)

Outputs:
  - models_v2/asof_comparison/asof_results_v20_domain_residual.csv
  - models_v2/asof_comparison/asof_predictions_v20_domain_residual.csv
  - models_v2/asof_comparison/asof_results_v20_domain_residual_filtered.csv
  - reports/asof_v20_domain_residual_results.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

INPUT = COMP_DIR / "asof_predictions_v18_peer_transfer.csv"
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
MIN_LOCAL_TRAIN_ROWS = 20
MIN_LOCAL_TRAIN_YEARS = 2

SCENARIOS = ["sunflower", "wheat_combined", "all_crops"]

TARGET_AND_LEAKAGE_COLUMNS = {
    "target_yield_t_ha",
    "target_source",
    "unit_fix_applied",
    "physical_t_ha",
    "productivity_t_ha",
    "prod_fact_t_ha",
    "target_v17",
    "target_v17_source",
    "target_v17_conflict",
    "cropwise_asof_t_ha",
    "estimate_history_dict",
    "history_item_id",
    "field_name",
    "prod_crop_ru",
    "sowing_date",
    "harvesting_date",
}

BASE_ID_COLUMNS = {"year"}

BASE_CAT_COLS = ["standard_name", "crop_id", "prev_crop_id", "rotation_pair"]

CORE_FEATURES = [
    "peer_pred",
    "standard_name",
    "crop_id",
    "prev_crop_id",
    "rotation_pair",
    "field_tillable_area",
    "ndvi_n_obs_asof",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_min_asof",
    "ndvi_std_asof",
    "ndvi_p75_asof",
    "ndvi_slope_asof",
    "ndvi_last_value_asof",
    "ndvi_integral_asof",
    "ndvi_peak_doy_asof",
    "ndvi_amplitude_asof",
    "ndvi_above_05_days_asof",
    "ndvi_anom_mean_asof",
    "ndvi_anom_max_asof",
    "ndvi_anom_min_asof",
    "ndvi_anom_last_value_asof",
    "ndvi_anom_last30d_mean_asof",
    "wx_temp_mean_to_asof",
    "wx_temp_max_to_asof",
    "wx_precip_sum_to_asof",
    "wx_srad_sum_to_asof",
    "wx_vpd_max_to_asof",
    "wx_vpd_mean_to_asof",
    "wx_vpd_days_high_to_asof",
    "wx_et0_sum_to_asof",
    "wx_water_balance_to_asof",
    "wx_gdd_sunflower_to_asof",
    "wx_hot_d30_to_asof",
    "wx_dry_days_to_asof",
    "wx_drought_run_to_asof",
    "wx_heat_run_to_asof",
    "sowing_date_valid",
    "days_after_sowing_asof",
    "sowing_doy",
    "gdd_from_sowing_asof",
]


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100.0),
    }


def local_dataset_path(tag: str) -> Path:
    preferred = DATA_PROCESSED / f"ml_dataset_clean_v17_v15_rates_clean_asof_{tag}.csv"
    if preferred.exists():
        return preferred
    return DATA_PROCESSED / f"ml_dataset_clean_v17_v12_clean_asof_{tag}.csv"


def load_local_with_peer(scenario: str, tag: str) -> pd.DataFrame:
    pred = pd.read_csv(INPUT)
    pred = pred[(pred["scenario"] == scenario) & (pred["asof_tag"] == tag)].copy()
    pred = pred.rename(columns={"pred": "peer_pred"})
    pred = pred[[
        "field_id",
        "year",
        "standard_name",
        "peer_pred",
        "cropwise_asof_t_ha",
        "naive_crop_mean",
    ]]

    local = pd.read_csv(local_dataset_path(tag))
    local = local.merge(
        pred,
        on=["field_id", "year", "standard_name"],
        how="inner",
        suffixes=("", "_from_peer"),
    )
    local["scenario"] = scenario
    local["asof_tag"] = tag
    return local.reset_index(drop=True)


def feature_columns(df: pd.DataFrame, mode: str) -> tuple[list[str], list[str]]:
    if mode == "core":
        cols = [c for c in CORE_FEATURES if c in df.columns]
    elif mode in {"all_no_field", "all_with_field"}:
        excluded = set(TARGET_AND_LEAKAGE_COLUMNS) | BASE_ID_COLUMNS | {
            "scenario",
            "asof_tag",
            "naive_crop_mean",
            "peer_pred_from_peer",
        }
        cols = [c for c in df.columns if c not in excluded]
        if "peer_pred" not in cols:
            cols.append("peer_pred")
        if mode == "all_no_field" and "field_id" in cols:
            cols.remove("field_id")
    else:
        raise ValueError(mode)

    # Drop columns with object/string dates or duplicated benchmark columns.
    cols = [c for c in cols if not c.endswith("_from_peer")]
    cat_cols = [c for c in BASE_CAT_COLS if c in cols]
    if mode == "all_with_field" and "field_id" in cols:
        cat_cols.append("field_id")
    return cols, cat_cols


def prepare_X(df: pd.DataFrame, cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for col in cat_cols:
        X[col] = X[col].apply(lambda v: "missing" if pd.isna(v) else str(v))
    for col in [c for c in cols if c not in cat_cols]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def residual_ridge(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> np.ndarray:
    cols, cat_cols = feature_columns(train, mode)
    X_train = prepare_X(train, cols, cat_cols)
    X_test = prepare_X(test, cols, cat_cols)
    y_res = train["target_yield_t_ha"] - train["peer_pred"]

    num_cols = [c for c in cols if c not in cat_cols]
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=10.0))])
    model.fit(X_train, y_res)
    return test["peer_pred"].values + model.predict(X_test)


def residual_catboost(train: pd.DataFrame, test: pd.DataFrame, mode: str, *, depth: int, l2: float) -> np.ndarray:
    cols, cat_cols = feature_columns(train, mode)
    X_train = prepare_X(train, cols, cat_cols)
    X_test = prepare_X(test, cols, cat_cols)
    y_res = train["target_yield_t_ha"] - train["peer_pred"]
    cat_idx = [cols.index(c) for c in cat_cols]

    params = {
        "iterations": 350,
        "learning_rate": 0.04,
        "depth": depth,
        "l2_leaf_reg": l2,
        "loss_function": "RMSE",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 40,
    }
    rng = np.random.default_rng(42)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    val_n = max(5, int(len(X_train) * 0.2))
    val_idx, fit_idx = idx[:val_n], idx[val_n:]
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(X_train.iloc[fit_idx], y_res.iloc[fit_idx], cat_features=cat_idx),
        eval_set=Pool(X_train.iloc[val_idx], y_res.iloc[val_idx], cat_features=cat_idx),
    )
    return test["peer_pred"].values + model.predict(X_test)


def residual_mean(train: pd.DataFrame, test: pd.DataFrame, by_crop: bool) -> np.ndarray:
    res = train["target_yield_t_ha"] - train["peer_pred"]
    if not by_crop:
        return test["peer_pred"].values + float(res.mean())
    train2 = train.assign(residual=res)
    global_res = float(train2["residual"].mean())
    by = train2.groupby("standard_name")["residual"].mean()
    return test["peer_pred"].values + test["standard_name"].map(by).fillna(global_res).astype(float).values


def calibrated_oof(df: pd.DataFrame, method: str) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for test_year in sorted(df["year"].dropna().unique()):
        test_mask = df["year"].values == test_year
        train = df[df["year"] < test_year].dropna(subset=["target_yield_t_ha", "peer_pred"]).copy()
        test = df[test_mask].dropna(subset=["peer_pred"]).copy()
        if test.empty:
            continue
        if method == "peer_raw" or len(train) < MIN_LOCAL_TRAIN_ROWS or train["year"].nunique() < MIN_LOCAL_TRAIN_YEARS:
            p = test["peer_pred"].values
        else:
            try:
                if method == "residual_global":
                    p = residual_mean(train, test, by_crop=False)
                elif method == "residual_crop":
                    p = residual_mean(train, test, by_crop=True)
                elif method == "ridge_core":
                    p = residual_ridge(train, test, mode="core")
                elif method == "ridge_all_no_field":
                    p = residual_ridge(train, test, mode="all_no_field")
                elif method == "cat_core_d2":
                    p = residual_catboost(train, test, mode="core", depth=2, l2=50.0)
                elif method == "cat_core_d3":
                    p = residual_catboost(train, test, mode="core", depth=3, l2=80.0)
                elif method == "cat_all_no_field":
                    p = residual_catboost(train, test, mode="all_no_field", depth=2, l2=100.0)
                elif method == "cat_all_with_field":
                    p = residual_catboost(train, test, mode="all_with_field", depth=2, l2=120.0)
                else:
                    raise ValueError(method)
            except Exception:
                p = test["peer_pred"].values
        pred[test.index.values] = p
    return pred


METHODS = [
    "peer_raw",
    "residual_global",
    "residual_crop",
    "ridge_core",
    "ridge_all_no_field",
    "cat_core_d2",
    "cat_core_d3",
    "cat_all_no_field",
    "cat_all_with_field",
]


def evaluate(df: pd.DataFrame, scenario: str, tag: str) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    for method in METHODS:
        p = calibrated_oof(df, method)
        cmp_df = df.assign(v20_pred=p).dropna(subset=["v20_pred"])
        if cmp_df.empty:
            continue
        ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["v20_pred"])
        cw = metrics(cmp_df["target_yield_t_ha"], cmp_df["cropwise_asof_t_ha"])
        nv = metrics(cmp_df["target_yield_t_ha"], cmp_df["naive_crop_mean"])
        rows.append({
            "scenario": scenario,
            "asof_tag": tag,
            "asof_label": ASOF_LABELS[tag],
            "method": method,
            "n": len(cmp_df),
            "mape": ml["mape"],
            "mae": ml["mae"],
            "rmse": ml["rmse"],
            "r2": ml["r2"],
            "cw_mape": cw["mape"],
            "cw_mae": cw["mae"],
            "cw_rmse": cw["rmse"],
            "cw_r2": cw["r2"],
            "naive_mape": nv["mape"],
            "naive_mae": nv["mae"],
            "naive_rmse": nv["rmse"],
            "naive_r2": nv["r2"],
        })
        keep_cols = [
            "scenario",
            "asof_tag",
            "field_id",
            "year",
            "standard_name",
            "target_yield_t_ha",
            "peer_pred",
            "cropwise_asof_t_ha",
            "naive_crop_mean",
            "v20_pred",
        ]
        keep = cmp_df[keep_cols].copy()
        keep["method"] = method
        preds.append(keep)
    return rows, preds


def filtered_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    filters = {
        "all": pred_df.index == pred_df.index,
        "target_ge_1": pred_df["target_yield_t_ha"] >= 1.0,
        "year_ge_2021": pred_df["year"] >= 2021,
        "target_ge_1_year_ge_2021": (pred_df["target_yield_t_ha"] >= 1.0) & (pred_df["year"] >= 2021),
    }
    rows = []
    for filter_name, mask in filters.items():
        cur = pred_df[mask].copy()
        for (scenario, tag, method), sub in cur.groupby(["scenario", "asof_tag", "method"]):
            if len(sub) < 2:
                continue
            ml = metrics(sub["target_yield_t_ha"], sub["v20_pred"])
            cw = metrics(sub["target_yield_t_ha"], sub["cropwise_asof_t_ha"])
            nv = metrics(sub["target_yield_t_ha"], sub["naive_crop_mean"])
            rows.append({
                "filter": filter_name,
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": ASOF_LABELS[tag],
                "method": method,
                "n": len(sub),
                "mape": ml["mape"],
                "mae": ml["mae"],
                "rmse": ml["rmse"],
                "r2": ml["r2"],
                "cw_mape": cw["mape"],
                "cw_mae": cw["mae"],
                "cw_rmse": cw["rmse"],
                "cw_r2": cw["r2"],
                "naive_mape": nv["mape"],
                "naive_mae": nv["mae"],
                "naive_rmse": nv["rmse"],
                "naive_r2": nv["r2"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 80)
    print("v20 DOMAIN-ADAPTED RESIDUAL")
    print("=" * 80)
    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ["07_01", "08_01", "09_01"]:
            df = load_local_with_peer(scenario, tag)
            part_rows, part_preds = evaluate(df, scenario, tag)
            rows.extend(part_rows)
            preds.extend(part_preds)
            sub = pd.DataFrame(part_rows)
            best_mape = sub.sort_values("mape").iloc[0]
            best_r2 = sub.sort_values("r2", ascending=False).iloc[0]
            print(
                f"  {tag}: best_mape={best_mape['method']} {best_mape['mape']:.1f}% "
                f"R2={best_mape['r2']:.3f}; best_r2={best_r2['method']} "
                f"{best_r2['r2']:.3f} MAPE={best_r2['mape']:.1f}%; "
                f"CW={best_mape['cw_mape']:.1f}% R2={best_mape['cw_r2']:.3f}"
            )

    summary = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    filt = filtered_summary(pred_df)

    out_results = COMP_DIR / "asof_results_v20_domain_residual.csv"
    out_preds = COMP_DIR / "asof_predictions_v20_domain_residual.csv"
    out_filtered = COMP_DIR / "asof_results_v20_domain_residual_filtered.csv"
    summary.to_csv(out_results, index=False)
    pred_df.to_csv(out_preds, index=False)
    filt.to_csv(out_filtered, index=False)

    best = summary.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"], as_index=False).first()
    best_filtered = (
        filt[filt["filter"].isin(["target_ge_1", "target_ge_1_year_ge_2021"])]
        .sort_values(["filter", "scenario", "asof_tag", "mape"])
        .groupby(["filter", "scenario", "asof_tag"], as_index=False)
        .first()
    )
    md = ["# v20 domain-adapted residual results\n\n"]
    md.append("Residual model: `target - peer_pred = f(local_features)`, trained on local past years only.\n\n")
    md.append("## Best method by MAPE (all rows)\n\n")
    md.append(best.round(4).to_markdown(index=False))
    md.append("\n\n## Best method under sensitivity filters\n\n")
    md.append(best_filtered.round(4).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(4).to_markdown(index=False))
    md.append("\n")
    out_md = REPORTS / "asof_v20_domain_residual_results.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print(f"\nSaved {out_results}")
    print(f"Saved {out_preds}")
    print(f"Saved {out_filtered}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
