"""
Sprint 2.1 + 2.2 — CatBoost on clean dataset v2.

Training scenarios:
  - all_crops          (n=441)
  - sunflower          (n=286)
  - wheat_combined     (wheat_spring + wheat_winter, n=106)

Validation:
  - Walk-Forward CV by year:
      For each test_year T from 2017 to 2025:
        train on rows where year < T, with at least MIN_TRAIN_YEARS unique years.
  - Final summary: out-of-fold predictions, per-fold metrics, aggregate metrics.

Per-fold metrics: R², RMSE (т/га), MAE (т/га), MAPE (%).

Cat features: crop_id, prev_crop_id, field_id (treated as Pool cat_features).

Outputs (per scenario):
  - models_v2/catboost_clean_<scenario>/oof.csv
  - models_v2/catboost_clean_<scenario>/fold_metrics.csv
  - models_v2/catboost_clean_<scenario>/feature_importance.csv
  - models_v2/catboost_clean_<scenario>/summary.json
  - models_v2/catboost_clean_<scenario>/predictions_scatter.png
  - reports/figures/catboost_<scenario>_oof_scatter.png
  - reports/figures/catboost_<scenario>_feature_importance.png
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
MODELS_V2.mkdir(parents=True, exist_ok=True)

DATASET = DATA_PROCESSED / "ml_dataset_clean_v2.csv"

CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id"]
NON_FEATURE = [
    "field_id",       # used as cat below
    "year",
    "target_yield_t_ha",
    "target_source",
    "unit_fix_applied",
    "standard_name",
]
# we KEEP field_id as a categorical feature (it's identity of a specific field — fertility prior)

MIN_TRAIN_YEARS = 4
MIN_TRAIN_ROWS = 30
WALK_FORWARD_TEST_YEARS = list(range(2017, 2026))  # inclusive 2017..2025


CATBOOST_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
    verbose=False,
)


@dataclass
class Scenario:
    name: str
    crop_filter: list[str] | None  # standard_name list, None = all crops


SCENARIOS: list[Scenario] = [
    Scenario(name="all_crops", crop_filter=None),
    Scenario(name="sunflower", crop_filter=["sunflower"]),
    Scenario(name="wheat_combined", crop_filter=["wheat_spring", "wheat_winter"]),
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(
        columns={"id": "crop_id"}
    )
    df = df.merge(crops, on="crop_id", how="left")
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    feat_cols = feat_cols + ["field_id"]  # add field_id as feature (it's removed from NON_FEATURE for X)
    feat_cols = list(dict.fromkeys(feat_cols))  # dedup keep order

    X = df[feat_cols].copy()

    # Coerce categorical features to clean strings — CatBoost requires string/int categoricals,
    # NaNs must be replaced.
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].apply(lambda v: "missing" if pd.isna(v) else str(int(float(v))))

    # Numeric coercion for the rest; keep NaN (CatBoost handles them natively for numeric).
    num_cols = [c for c in feat_cols if c not in CAT_FEATURES]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    y = df["target_yield_t_ha"].astype(float)
    return X, y, feat_cols


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def fit_predict_one(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_idx: list[int],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, CatBoostRegressor]:
    """
    Fit CatBoost with internal validation split (last val_frac of train rows) for early stopping.
    Returns predictions for X_test and the fitted model.
    """
    n = len(X_train)
    if n < 30:
        # Tiny train — skip val split, fewer iters
        model = CatBoostRegressor(**{**CATBOOST_PARAMS, "iterations": 200, "od_wait": 30})
        model.fit(Pool(X_train, y_train, cat_features=cat_idx))
    else:
        # Reproducible last-fraction val (we already chronologically split outer fold by year).
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_n = max(5, int(n * val_frac))
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]
        train_pool = Pool(X_train.iloc[tr_idx], y_train.iloc[tr_idx], cat_features=cat_idx)
        val_pool = Pool(X_train.iloc[val_idx], y_train.iloc[val_idx], cat_features=cat_idx)
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool)
    pred = model.predict(X_test)
    return pred, model


def walk_forward_evaluate(df: pd.DataFrame) -> dict:
    X, y, feat_cols = prepare_xy(df)
    cat_idx = [feat_cols.index(c) for c in CAT_FEATURES if c in feat_cols]
    years = df["year"].values

    oof = np.full(len(df), np.nan)
    fold_rows = []
    feature_importance_acc = pd.Series(0.0, index=feat_cols)
    n_models = 0

    last_model = None

    for test_year in WALK_FORWARD_TEST_YEARS:
        test_mask = years == test_year
        train_mask = years < test_year
        if test_mask.sum() == 0:
            continue
        unique_train_years = pd.unique(years[train_mask])
        if (
            train_mask.sum() < MIN_TRAIN_ROWS
            or len(unique_train_years) < MIN_TRAIN_YEARS
        ):
            continue

        X_train = X.iloc[train_mask]
        y_train = y.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_test = y.iloc[test_mask]

        pred, model = fit_predict_one(X_train, y_train, X_test, cat_idx)
        oof[test_mask] = pred
        last_model = model

        fold_rows.append(
            {
                "test_year": int(test_year),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "r2": float(r2_score(y_test, pred)) if len(y_test) >= 2 else float("nan"),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": float(mean_absolute_error(y_test, pred)),
                "mape": mape(y_test, pred),
            }
        )
        # Aggregate feature importance (averaged later)
        try:
            fi = pd.Series(model.get_feature_importance(), index=feat_cols)
            feature_importance_acc = feature_importance_acc.add(fi, fill_value=0.0)
            n_models += 1
        except Exception:
            pass

    feature_importance = feature_importance_acc / max(1, n_models)

    fold_metrics = pd.DataFrame(fold_rows)
    valid_mask = ~np.isnan(oof)
    yv = y.values[valid_mask]
    pv = oof[valid_mask]
    overall = {
        "n_total": int(len(df)),
        "n_oof": int(valid_mask.sum()),
        "oof_r2": float(r2_score(yv, pv)) if valid_mask.sum() >= 2 else float("nan"),
        "oof_rmse": float(np.sqrt(mean_squared_error(yv, pv))) if valid_mask.sum() >= 2 else float("nan"),
        "oof_mae": float(mean_absolute_error(yv, pv)) if valid_mask.sum() >= 2 else float("nan"),
        "oof_mape": mape(yv, pv),
        "fold_r2_mean": float(np.nanmean(fold_metrics["r2"])) if len(fold_metrics) else float("nan"),
        "fold_r2_std": float(np.nanstd(fold_metrics["r2"])) if len(fold_metrics) else float("nan"),
        "fold_mape_mean": float(np.nanmean(fold_metrics["mape"])) if len(fold_metrics) else float("nan"),
        "fold_mape_std": float(np.nanstd(fold_metrics["mape"])) if len(fold_metrics) else float("nan"),
        "n_folds": int(len(fold_metrics)),
    }

    return {
        "fold_metrics": fold_metrics,
        "feature_importance": feature_importance.sort_values(ascending=False),
        "oof_pred": oof,
        "y_true": y.values,
        "df_keys": df[["field_id", "year", "crop_id", "standard_name"]].reset_index(drop=True),
        "overall": overall,
        "last_model": last_model,
    }


def plot_oof_scatter(y_true: np.ndarray, oof_pred: np.ndarray, name: str, overall: dict) -> Path:
    mask = ~np.isnan(oof_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[mask], oof_pred[mask], alpha=0.5, s=22, edgecolor="white")
    lo = min(np.nanmin(y_true[mask]), np.nanmin(oof_pred[mask]))
    hi = max(np.nanmax(y_true[mask]), np.nanmax(oof_pred[mask]))
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="y=x")
    plt.xlabel("Actual yield, т/га")
    plt.ylabel("OOF prediction, т/га")
    plt.title(
        f"CatBoost OOF — {name}\n"
        f"n={overall['n_oof']}, R²={overall['oof_r2']:.3f}, RMSE={overall['oof_rmse']:.2f} т/га, MAPE={overall['oof_mape']:.1f}%"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = FIGS / f"catboost_{name}_oof_scatter.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def plot_feature_importance(fi: pd.Series, name: str, top_n: int = 20) -> Path:
    top = fi.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, len(top) * 0.32)))
    plt.barh(top.index, top.values, color="#3a86ff")
    plt.xlabel("Mean importance")
    plt.title(f"CatBoost feature importance — {name}  (top {top_n})")
    plt.tight_layout()
    p = FIGS / f"catboost_{name}_feature_importance.png"
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def run_scenario(df: pd.DataFrame, sc: Scenario) -> dict:
    print("\n" + "=" * 80)
    print(f"SCENARIO: {sc.name}")
    print("=" * 80)
    if sc.crop_filter is not None:
        sub = df[df["standard_name"].isin(sc.crop_filter)].copy()
    else:
        sub = df.copy()
    sub = sub.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)
    print(f"  Rows: {len(sub)}, years: {sorted(sub['year'].unique().tolist())}")
    if len(sub) < 50:
        print("  → too few rows, skipping per-fold")
        return {}

    res = walk_forward_evaluate(sub)
    overall = res["overall"]
    print(
        f"  OOF R²={overall['oof_r2']:.3f}  RMSE={overall['oof_rmse']:.3f}  "
        f"MAE={overall['oof_mae']:.3f}  MAPE={overall['oof_mape']:.2f}%  "
        f"(folds={overall['n_folds']})"
    )

    out_dir = MODELS_V2 / f"catboost_clean_{sc.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_df = res["df_keys"].copy()
    oof_df["target_yield_t_ha"] = res["y_true"]
    oof_df["oof_pred_t_ha"] = res["oof_pred"]
    oof_df.to_csv(out_dir / "oof.csv", index=False)

    res["fold_metrics"].to_csv(out_dir / "fold_metrics.csv", index=False)
    res["feature_importance"].to_csv(out_dir / "feature_importance.csv", header=["importance"])
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    plot_oof_scatter(res["y_true"], res["oof_pred"], sc.name, overall)
    plot_feature_importance(res["feature_importance"], sc.name, top_n=20)

    return overall


def main():
    print("=" * 80)
    print("TRAIN CLEAN CATBOOST — Sprint 2")
    print("=" * 80)
    df = load_dataset()
    print(f"Loaded {len(df)} rows × {len(df.columns)} cols")

    summary_rows = []
    for sc in SCENARIOS:
        overall = run_scenario(df, sc)
        if overall:
            row = {"scenario": sc.name, **overall}
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(MODELS_V2 / "catboost_clean_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved {MODELS_V2 / 'catboost_clean_summary.csv'}")


if __name__ == "__main__":
    main()
