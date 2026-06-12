"""
Train v18 models on the large productivity_estimate_peers dataset.

This is intentionally separate from the v9/v17 trainer because peer rows do not
have our internal field_id / Cropwise benchmark history. We evaluate:
  - naive walk-forward baselines;
  - CatBoost direct target prediction;
  - CatBoost residual prediction over train-fold crop means.

Validation:
  - walk-forward by year;
  - no row from the target year is visible in training.

Outputs:
  models_v2/asof_comparison/asof_results_v18_peers.csv
  reports/asof_v18_peers_results.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

ASOF_TAGS = ["07_01", "08_01", "09_01"]
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
WALK_FORWARD_TEST_YEARS = list(range(2014, 2026))
MIN_TRAIN_YEARS = 3
MIN_TRAIN_ROWS = 500

SCENARIOS = [
    ("sunflower", ["sunflower"]),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("all_crops", None),
]

VARIANTS = ["strict", "plus"]

NON_FEATURE = {
    "peer_row_id",
    "year",
    "target_yield_t_ha",
    "target_source",
    "asof_tag",
    "standard_name",
}

CAT_FEATURES = [
    "crop",
    "variety",
    "previous_crop_name",
    "previous_crop_productivity_estimate_crop_name",
    "previous_crop_standard_name",
]

PARAM_SETS: dict[str, dict] = {
    "fast_l10": dict(iterations=450, learning_rate=0.06, depth=5, l2_leaf_reg=10.0, loss_function="RMSE"),
    "shallow_l30": dict(iterations=650, learning_rate=0.045, depth=4, l2_leaf_reg=30.0, loss_function="RMSE"),
    "mae_l20": dict(iterations=550, learning_rate=0.05, depth=4, l2_leaf_reg=20.0, loss_function="MAE"),
}


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


def load_dataset(variant: str, tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    path = DATA_PROCESSED / f"ml_dataset_v18_peers_{variant}_asof_{tag}.csv"
    df = pd.read_csv(path)
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    return df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)


def prepare_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feat_cols].copy()
    cat_in_x = [c for c in CAT_FEATURES if c in X.columns]
    for col in cat_in_x:
        X[col] = X[col].fillna("missing").astype(str)
    for col in [c for c in feat_cols if c not in cat_in_x]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    cat_idx = [feat_cols.index(c) for c in cat_in_x]
    y = df["target_yield_t_ha"].astype(float)
    return X, y, feat_cols, cat_idx


def model_params(params: dict) -> dict:
    return {
        **params,
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 60,
    }


def walk_forward_model_oof(
    df: pd.DataFrame,
    params: dict,
    *,
    residual: bool = False,
) -> np.ndarray:
    X, y, _, cat_idx = prepare_xy(df)
    years = df["year"].values
    crops = df["standard_name"].values
    oof = np.full(len(df), np.nan)

    for test_year in WALK_FORWARD_TEST_YEARS:
        test_mask = years == test_year
        train_mask = years < test_year
        if test_mask.sum() == 0:
            continue
        if train_mask.sum() < MIN_TRAIN_ROWS or len(pd.unique(years[train_mask])) < MIN_TRAIN_YEARS:
            continue

        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_fit = y_train.copy()

        crop_mean = None
        global_mean = float(y_train.mean())
        if residual:
            train_crops = pd.Series(crops[train_mask], index=y_train.index)
            crop_mean = y_train.groupby(train_crops).mean()
            y_fit = y_train - train_crops.map(crop_mean).fillna(global_mean).astype(float)

        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(50, int(len(X_train) * 0.15))
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        model = CatBoostRegressor(**model_params(params))
        model.fit(
            Pool(X_train.iloc[train_idx], y_fit.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_fit.iloc[val_idx], cat_features=cat_idx),
        )
        pred = model.predict(X_test)
        if residual and crop_mean is not None:
            test_crops = pd.Series(crops[test_mask], index=X_test.index)
            pred = pred + test_crops.map(crop_mean).fillna(global_mean).astype(float).values
        oof[test_mask] = pred

    return oof


def naive_oof(df: pd.DataFrame, mode: str) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for test_year in WALK_FORWARD_TEST_YEARS:
        test_mask = df["year"].values == test_year
        train = df[df["year"] < test_year]
        if test_mask.sum() == 0 or len(train) < MIN_TRAIN_ROWS:
            continue
        global_mean = float(train["target_yield_t_ha"].mean())
        if mode == "global_mean":
            pred[test_mask] = global_mean
        elif mode == "crop_mean":
            means = train.groupby("standard_name")["target_yield_t_ha"].mean()
            pred[test_mask] = df.loc[test_mask, "standard_name"].map(means).fillna(global_mean).values
        elif mode == "crop_year_trend":
            # Conservative trend baseline: crop mean from the previous 3 train years.
            recent_years = sorted(train["year"].unique())[-3:]
            recent = train[train["year"].isin(recent_years)]
            means = recent.groupby("standard_name")["target_yield_t_ha"].mean()
            fallback = recent["target_yield_t_ha"].mean()
            pred[test_mask] = df.loc[test_mask, "standard_name"].map(means).fillna(fallback).values
        else:
            raise ValueError(mode)
    return pred


def eval_one(variant: str, tag: str, scenario: str, crop_filter: list[str] | None) -> list[dict[str, object]]:
    df = load_dataset(variant, tag, crop_filter)
    rows: list[dict[str, object]] = []
    if len(df) < MIN_TRAIN_ROWS:
        return rows

    for mode in ["global_mean", "crop_mean", "crop_year_trend"]:
        pred = naive_oof(df, mode)
        cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
        m = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
        rows.append({
            "scenario": scenario,
            "asof_tag": tag,
            "asof_label": ASOF_LABELS[tag],
            "variant": variant,
            "model": mode,
            "model_type": "naive",
            "n": len(cmp_df),
            **m,
        })

    for params_name, params in PARAM_SETS.items():
        for residual in [False, True]:
            model_name = f"catboost_{params_name}" + ("_residual" if residual else "")
            pred = walk_forward_model_oof(df, params, residual=residual)
            cmp_df = df.assign(pred=pred).dropna(subset=["pred"])
            m = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
            rows.append({
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": ASOF_LABELS[tag],
                "variant": variant,
                "model": model_name,
                "model_type": "ml",
                "n": len(cmp_df),
                **m,
            })
    return rows


def main() -> None:
    print("=" * 80)
    print("TRAIN v18 — PRODUCTIVITY ESTIMATE PEERS")
    print("=" * 80)
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        print(f"\nVARIANT: {variant}")
        for scenario, crop_filter in SCENARIOS:
            print(f"  {scenario}")
            for tag in ASOF_TAGS:
                part = eval_one(variant, tag, scenario, crop_filter)
                rows.extend(part)
                sub = pd.DataFrame(part)
                if sub.empty:
                    print(f"    {tag}: no result")
                    continue
                best_ml = sub[sub["model_type"] == "ml"].sort_values("mape").iloc[0]
                best_naive = sub[sub["model_type"] == "naive"].sort_values("mape").iloc[0]
                print(
                    f"    {tag}: ML={best_ml['model']} MAPE={best_ml['mape']:.1f}% "
                    f"R2={best_ml['r2']:.3f}; naive={best_naive['model']} "
                    f"MAPE={best_naive['mape']:.1f}% R2={best_naive['r2']:.3f}; "
                    f"n={int(best_ml['n'])}"
                )

    summary = pd.DataFrame(rows)
    out_csv = COMP_DIR / "asof_results_v18_peers.csv"
    summary.to_csv(out_csv, index=False)

    best = (
        summary[summary["model_type"] == "ml"]
        .sort_values(["variant", "scenario", "asof_tag", "mape"])
        .groupby(["variant", "scenario", "asof_tag"], as_index=False)
        .first()
    )
    best_naive = (
        summary[summary["model_type"] == "naive"]
        .sort_values(["variant", "scenario", "asof_tag", "mape"])
        .groupby(["variant", "scenario", "asof_tag"], as_index=False)
        .first()
        .rename(columns={"model": "best_naive", "mape": "naive_mape", "mae": "naive_mae", "rmse": "naive_rmse", "r2": "naive_r2"})
    )
    joined = best.merge(
        best_naive[["variant", "scenario", "asof_tag", "best_naive", "naive_mape", "naive_mae", "naive_rmse", "naive_r2"]],
        on=["variant", "scenario", "asof_tag"],
        how="left",
    )

    md = ["# v18 peers results\n\n"]
    md.append("Walk-forward by year on `productivity_estimate_peers.csv`.\n\n")
    md.append("## Best ML vs best naive baseline\n\n")
    md.append(joined.round(4).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(4).to_markdown(index=False))
    md.append("\n")
    out_md = REPORTS / "asof_v18_peers_results.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print("\nFINAL BEST")
    print(joined.round(4).to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
