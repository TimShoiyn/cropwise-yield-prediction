"""
External transfer test:
  train on v18 peers (large peer dataset),
  predict our 30-field v17 dataset,
  compare against Cropwise on the same rows.

For each test year in our data, training uses only peer rows with year < test_year.
This keeps the same walk-forward logic and avoids future leakage.

Outputs:
  models_v2/asof_comparison/asof_results_v18_peer_transfer.csv
  reports/asof_v18_peer_transfer_results.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

ASOF_TAGS = ["07_01", "08_01", "09_01"]
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}

SCENARIOS = [
    ("sunflower", ["sunflower"]),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("all_crops", None),
]

PARAMS = {
    "fast_l10": dict(iterations=450, learning_rate=0.06, depth=5, l2_leaf_reg=10.0, loss_function="RMSE"),
    "shallow_l30": dict(iterations=650, learning_rate=0.045, depth=4, l2_leaf_reg=30.0, loss_function="RMSE"),
    "mae_l20": dict(iterations=550, learning_rate=0.05, depth=4, l2_leaf_reg=20.0, loss_function="MAE"),
}

# Use the strict-v18 winners as a conservative first transfer policy.
POLICY = {
    ("sunflower", "07_01"): ("mae_l20", True),
    ("sunflower", "08_01"): ("mae_l20", False),
    ("sunflower", "09_01"): ("mae_l20", False),
    ("wheat_combined", "07_01"): ("fast_l10", True),
    ("wheat_combined", "08_01"): ("shallow_l30", True),
    ("wheat_combined", "09_01"): ("mae_l20", False),
    ("all_crops", "07_01"): ("mae_l20", True),
    ("all_crops", "08_01"): ("mae_l20", False),
    ("all_crops", "09_01"): ("mae_l20", False),
}

FEATURE_COLS = [
    "crop",
    "variety",
    "previous_crop_name",
    "previous_crop_productivity_estimate_crop_name",
    "previous_crop_standard_name",
    "area",
    "ndvi_n_obs_asof",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_min_asof",
    "ndvi_std_asof",
    "ndvi_p25_asof",
    "ndvi_p75_asof",
    "ndvi_last_value_asof",
    "ndvi_last_doy_asof",
    "ndvi_peak_doy_asof",
    "ndvi_amplitude_asof",
    "ndvi_slope_asof",
    "ndvi_integral_asof",
    "ndvi_last30d_mean_asof",
    "ndvi_above_05_n_asof",
    "sowing_known_asof",
    "sowing_doy",
    "days_after_sowing_asof",
]
CAT_COLS = [
    "crop",
    "variety",
    "previous_crop_name",
    "previous_crop_productivity_estimate_crop_name",
    "previous_crop_standard_name",
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


def model_params(params: dict) -> dict:
    return {
        **params,
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 60,
    }


def load_peers(tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v18_peers_strict_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    return df.reset_index(drop=True)


def prev_crop_lookup() -> dict[int, str]:
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    return dict(zip(crops["crop_id"], crops["standard_name"]))


def load_ours_as_v18(tag: str, crop_filter: list[str] | None, cw: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_clean_v17_v12_clean_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = attach_cropwise(df, tag, cw)
    crop_map = prev_crop_lookup()

    out = pd.DataFrame({
        "field_id": df["field_id"],
        "year": df["year"],
        "target_yield_t_ha": df["target_yield_t_ha"],
        "cropwise_asof_t_ha": df["cropwise_asof_t_ha"],
        "standard_name": df["standard_name"],
        "crop": df["standard_name"].fillna("missing").astype(str),
        "variety": "missing",
        "previous_crop_name": df.get("prev_crop_id", pd.Series(index=df.index, dtype=float)).map(crop_map).fillna("missing").astype(str),
        "previous_crop_productivity_estimate_crop_name": df.get("prev_crop_id", pd.Series(index=df.index, dtype=float)).map(crop_map).fillna("missing").astype(str),
        "previous_crop_standard_name": df.get("prev_crop_id", pd.Series(index=df.index, dtype=float)).map(crop_map).fillna("missing").astype(str),
        "area": df["field_tillable_area"],
        "ndvi_n_obs_asof": df["ndvi_n_obs_asof"],
        "ndvi_mean_asof": df["ndvi_mean_asof"],
        "ndvi_max_asof": df["ndvi_max_asof"],
        "ndvi_min_asof": df["ndvi_min_asof"],
        "ndvi_std_asof": df["ndvi_std_asof"],
        "ndvi_p25_asof": df["ndvi_p25_asof"],
        "ndvi_p75_asof": df["ndvi_p75_asof"],
        "ndvi_last_value_asof": df["ndvi_last_value_asof"],
        "ndvi_last_doy_asof": np.nan,
        "ndvi_peak_doy_asof": df["ndvi_peak_doy_asof"],
        "ndvi_amplitude_asof": df["ndvi_amplitude_asof"],
        "ndvi_slope_asof": df["ndvi_slope_asof"],
        "ndvi_integral_asof": df["ndvi_integral_asof"],
        "ndvi_last30d_mean_asof": df.get("ndvi_anom_last30d_mean_asof", np.nan),
        "ndvi_above_05_n_asof": df["ndvi_above_05_days_asof"],
        "sowing_known_asof": df["sowing_date_valid"].fillna(0),
        "sowing_doy": df["sowing_doy"],
        "days_after_sowing_asof": df["days_after_sowing_asof"],
    })
    return out.reset_index(drop=True)


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].fillna("missing").astype(str)
    for col in [c for c in FEATURE_COLS if c not in CAT_COLS]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def predict_transfer(
    peers: pd.DataFrame,
    ours: pd.DataFrame,
    params_name: str,
    residual: bool,
) -> np.ndarray:
    pred = np.full(len(ours), np.nan)
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_COLS]

    for test_year in sorted(ours["year"].unique()):
        test_mask = ours["year"].values == test_year
        train = peers[peers["year"] < test_year].copy()
        if len(train) < 500:
            continue
        test = ours[test_mask].copy()

        X_train = prepare_X(train)
        X_test = prepare_X(test)
        y_train = train["target_yield_t_ha"].astype(float)

        global_mean = float(y_train.mean())
        y_fit = y_train.copy()
        crop_mean = None
        if residual:
            crop_mean = y_train.groupby(train["standard_name"]).mean()
            y_fit = y_train - train["standard_name"].map(crop_mean).fillna(global_mean).astype(float)

        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(50, int(len(X_train) * 0.15))
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        model = CatBoostRegressor(**model_params(PARAMS[params_name]))
        model.fit(
            Pool(X_train.iloc[train_idx], y_fit.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_fit.iloc[val_idx], cat_features=cat_idx),
        )
        p = model.predict(X_test)
        if residual and crop_mean is not None:
            p = p + test["standard_name"].map(crop_mean).fillna(global_mean).astype(float).values
        pred[test_mask] = p
    return pred


def naive_crop_mean_transfer(peers: pd.DataFrame, ours: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(ours), np.nan)
    for test_year in sorted(ours["year"].unique()):
        test_mask = ours["year"].values == test_year
        train = peers[peers["year"] < test_year]
        if len(train) < 500:
            continue
        global_mean = float(train["target_yield_t_ha"].mean())
        means = train.groupby("standard_name")["target_yield_t_ha"].mean()
        pred[test_mask] = ours.loc[test_mask, "standard_name"].map(means).fillna(global_mean).values
    return pred


def main() -> None:
    print("=" * 80)
    print("EVALUATE v18 PEER TRANSFER ON OUR 30 FIELDS")
    print("=" * 80)
    cw = cropwise_table()
    rows = []
    pred_rows = []
    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            peers = load_peers(tag, crop_filter)
            ours = load_ours_as_v18(tag, crop_filter, cw)
            params_name, residual = POLICY[(scenario, tag)]
            pred = predict_transfer(peers, ours, params_name, residual)
            naive = naive_crop_mean_transfer(peers, ours)
            cmp_df = ours.assign(pred=pred, naive_crop_mean=naive).dropna(subset=["pred"])
            cmp_cw = cmp_df.dropna(subset=["cropwise_asof_t_ha"])
            ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
            cw_m = metrics(cmp_cw["target_yield_t_ha"], cmp_cw["cropwise_asof_t_ha"]) if len(cmp_cw) else {}
            nv = metrics(cmp_df["target_yield_t_ha"], cmp_df["naive_crop_mean"])
            row = {
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": ASOF_LABELS[tag],
                "n": len(cmp_df),
                "params": params_name,
                "residual": residual,
                "mape": ml["mape"],
                "mae": ml["mae"],
                "rmse": ml["rmse"],
                "r2": ml["r2"],
                "cw_mape": cw_m.get("mape", np.nan),
                "cw_mae": cw_m.get("mae", np.nan),
                "cw_rmse": cw_m.get("rmse", np.nan),
                "cw_r2": cw_m.get("r2", np.nan),
                "naive_mape": nv["mape"],
                "naive_mae": nv["mae"],
                "naive_rmse": nv["rmse"],
                "naive_r2": nv["r2"],
            }
            rows.append(row)
            tmp = cmp_df[["field_id", "year", "standard_name", "target_yield_t_ha", "pred", "cropwise_asof_t_ha", "naive_crop_mean"]].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)
            print(
                f"  {tag}: ML MAPE={row['mape']:.1f}% R2={row['r2']:.3f}; "
                f"CW={row['cw_mape']:.1f}% R2={row['cw_r2']:.3f}; "
                f"naive={row['naive_mape']:.1f}% R2={row['naive_r2']:.3f}; n={len(cmp_df)}"
            )

    summary = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v18_peer_transfer.csv"
    summary.to_csv(out_csv, index=False)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(COMP_DIR / "asof_predictions_v18_peer_transfer.csv", index=False)

    md = ["# v18 peer transfer results\n\n"]
    md.append("Train: v18 peers, with train years < test year. Test: our v17 30-field clean dataset.\n\n")
    md.append(summary.round(4).to_markdown(index=False))
    md.append("\n")
    out_md = REPORTS / "asof_v18_peer_transfer_results.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"\nSaved {out_csv}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
