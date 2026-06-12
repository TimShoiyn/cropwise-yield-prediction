"""
Iteration 2 - Step B: CatBoost experiments on full_extended + all_no_leak.

Запуск:
    python train_iteration2_catboost.py

Результат:
    - models/models_iteration2_catboost_full_extended.csv  (CV и train метрики по конфигурациям)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor, Pool


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_DIR = "data_processed"
OUTPUT_DIR = "models"

RANDOM_STATE = 42
CV_FOLDS = 5


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def prepare_full_extended_all_no_leak(df: pd.DataFrame):
    """
    Повторяет логику prepare_features(..., model_type='all', exclude_leaky=True)
    из train_baseline_model.py, но без sklearn-пайплайна.
    """
    leaky_features = ["ops_yield_t_ha"]

    exclude_cols = [
        "field_id",
        "year",
        "target_yield_t_ha",
        "crop_name",
        "ops_season_start",
        "ops_season_end",
    ]

    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    all_feature_cols = [c for c in all_feature_cols if c not in leaky_features]

    X = df[all_feature_cols].copy()
    y = df["target_yield_t_ha"].astype(float).copy()
    groups = df["year"].values

    # CatBoost сам умеет работать с пропусками в числовых фичах, поэтому не делаем импутацию.
    # crop_id / prev_crop_id будем передавать как категориальные и приводим их к строковому типу.
    feature_cols = X.columns.tolist()

    cat_cols = [c for c in feature_cols if c in ("crop_id", "prev_crop_id")]
    for c in cat_cols:
        # Приводим к строковому типу и ЯВНО заменяем NaN на строку "nan"
        col = X[c]
        col = col.astype("object")
        col = col.where(~col.isna(), other="nan")
        X[c] = col.astype(str)

    # Все остальные колонки должны быть числовыми: пытаемся привести к float,
    # любые строки (например, даты) превращаем в NaN.
    for c in feature_cols:
        if c in cat_cols:
            continue
        if not is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Индексы категориальных признаков (по позициям в X)
    cat_features = [i for i, c in enumerate(feature_cols) if c in cat_cols]

    return X, y, groups, feature_cols, cat_features


def evaluate_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    cat_features: list[int],
    params: dict,
    n_splits: int = CV_FOLDS,
) -> dict:
    """
    GroupKFold по годам + OOF-предсказания для CatBoost.
    """
    gkf = GroupKFold(n_splits=n_splits)
    y_arr = y.to_numpy(dtype=float)
    oof_pred = np.full_like(y_arr, fill_value=np.nan, dtype=float)

    fold_r2 = []
    fold_rmse = []
    fold_mae = []
    fold_mape = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_arr, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y_arr[train_idx], y_arr[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            **params,
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        y_pred = model.predict(val_pool)
        oof_pred[test_idx] = y_pred

        fold_r2.append(r2_score(y_val, y_pred))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        fold_mae.append(float(mean_absolute_error(y_val, y_pred)))
        fold_mape.append(_mape(y_val, y_pred))

        print(
            f"  Fold {fold_idx}: "
            f"R² = {fold_r2[-1]:.4f}, RMSE = {fold_rmse[-1]:.4f}, "
            f"MAPE = {fold_mape[-1]:.2f}%"
        )

    # Train на всём датасете (для train-метрик)
    full_pool = Pool(X, y_arr, cat_features=cat_features)
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        **params,
    )
    final_model.fit(full_pool)
    y_train_pred = final_model.predict(full_pool)

    return {
        "cv_r2_mean": float(np.nanmean(fold_r2)),
        "cv_r2_std": float(np.nanstd(fold_r2)),
        "cv_rmse_mean": float(np.nanmean(fold_rmse)),
        "cv_rmse_std": float(np.nanstd(fold_rmse)),
        "cv_mae_mean": float(np.nanmean(fold_mae)),
        "cv_mae_std": float(np.nanstd(fold_mae)),
        "cv_mape_mean": float(np.nanmean(fold_mape)),
        "cv_mape_std": float(np.nanstd(fold_mape)),
        "train_r2": float(r2_score(y_arr, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_arr, y_train_pred))),
        "train_mae": float(mean_absolute_error(y_arr, y_train_pred)),
        "train_mape": float(_mape(y_arr, y_train_pred)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_path = os.path.join(DATA_DIR, "ml_dataset_full_extended.csv")
    if not os.path.exists(full_path):
        print(f"❌ Не найден {full_path}")
        return

    df = pd.read_csv(full_path)
    if "target_yield_t_ha" not in df.columns:
        print("❌ В датасете нет target_yield_t_ha")
        return

    print("=== Iteration 2 / Step B: CatBoost on full_extended + all_no_leak ===")
    print(f"Файл: {full_path}")
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")
    print()

    X, y, groups, feature_cols, cat_features = prepare_full_extended_all_no_leak(df)
    print(f"Фич (всего): {len(feature_cols)}, cat_features idx: {cat_features}")
    print()

    # Сетка гиперпараметров (6 комбинаций)
    param_grid = [
        {"depth": 3, "l2_leaf_reg": 3},
        {"depth": 3, "l2_leaf_reg": 5},
        {"depth": 4, "l2_leaf_reg": 3},
        {"depth": 4, "l2_leaf_reg": 5},
        {"depth": 5, "l2_leaf_reg": 5},
        {"depth": 5, "l2_leaf_reg": 10},
    ]

    base_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_data_in_leaf": 10,
    }

    results = []

    for i, grid_params in enumerate(param_grid, start=1):
        params = {**base_params, **grid_params}

        print("-" * 80)
        print(
            f"Config {i}/{len(param_grid)}: "
            f"depth={params['depth']}, "
            f"l2_leaf_reg={params['l2_leaf_reg']}, "
            f"lr={params['learning_rate']}, "
            f"iters={params['iterations']}, "
            f"subsample={params['subsample']}"
        )

        res = evaluate_catboost_cv(X, y, groups, cat_features, params)

        print(
            f"  -> CV R² = {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}, "
            f"CV RMSE = {res['cv_rmse_mean']:.4f}, "
            f"CV MAPE = {res['cv_mape_mean']:.2f}%, "
            f"Train R² = {res['train_r2']:.4f}"
        )
        print()

        row = {
            "dataset_name": "full_extended",
            "model_type": "catboost_all_no_leak",
            "depth": params["depth"],
            "l2_leaf_reg": params["l2_leaf_reg"],
            "learning_rate": params["learning_rate"],
            "iterations": params["iterations"],
            "subsample": params["subsample"],
            "min_data_in_leaf": params["min_data_in_leaf"],
            "cv_r2_mean": res["cv_r2_mean"],
            "cv_r2_std": res["cv_r2_std"],
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_rmse_std": res["cv_rmse_std"],
            "cv_mae_mean": res["cv_mae_mean"],
            "cv_mae_std": res["cv_mae_std"],
            "cv_mape_mean": res["cv_mape_mean"],
            "cv_mape_std": res["cv_mape_std"],
            "train_r2": res["train_r2"],
            "train_rmse": res["train_rmse"],
            "train_mae": res["train_mae"],
            "train_mape": res["train_mape"],
        }
        results.append(row)

    df_res = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "models_iteration2_catboost_full_extended.csv")
    df_res.to_csv(out_path, index=False)
    print("-" * 80)
    print(f"💾 Результаты CatBoost сохранены: {out_path}")

    best_row = df_res.iloc[df_res["cv_r2_mean"].idxmax()]
    print()
    print("🏆 Лучшая CatBoost-конфигурация (по CV R²):")
    print(best_row.to_string())
    print()


if __name__ == "__main__":
    main()

