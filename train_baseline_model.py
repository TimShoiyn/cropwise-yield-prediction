"""
Baseline ML модель для прогноза урожайности.

Обучает 3 модели:
1. Только агрооперации + геометрия (без NDVI)
2. Только NDVI-фичи
3. Все признаки вместе

Использует GroupKFold по годам для cross-validation.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Always run relative to this file (fixes Windows cwd/path issues)
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data_processed"
OUTPUT_DIR = "models"
DATA_RAW_DIR = "data_raw"

# Датасеты для обучения
DATASETS = [
    {
        "name": "base_with_ndvi",
        "path": os.path.join(DATA_DIR, "ml_dataset_with_ndvi.csv"),
    },
    {
        "name": "ndvi_only_extended",
        "path": os.path.join(DATA_DIR, "ml_dataset_ndvi_only_extended.csv"),
    },
    {
        "name": "ops_ndvi_2021_2025",
        "path": os.path.join(DATA_DIR, "ml_dataset_ops_ndvi_2021_2025.csv"),
    },
    {
        "name": "full_extended",
        "path": os.path.join(DATA_DIR, "ml_dataset_full_extended.csv"),
    },
    {
        "name": "full_extended_cropwindow_v1",
        "path": os.path.join(DATA_DIR, "ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv"),
    },
]

# Параметры модели
N_ESTIMATORS = 100
MAX_DEPTH = 5
LEARNING_RATE = 0.1
RANDOM_STATE = 42
CV_FOLDS = 5

# Best regularized GBDT params (Iteration 2 winner)
BEST_REG_GBDT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 0.7,
    "learning_rate": 0.05,
}

# Crop filters (use crops.csv standard_name)
STANDARD_NAME_WHEAT_SPRING = "wheat_spring"
STANDARD_NAME_SUNFLOWER = "sunflower"


def _load_crops_standard_name_to_ids() -> dict[str, set[int]]:
    """
    Returns mapping: standard_name -> set(crop_id) from data_raw/crops.csv.
    Used to filter cropwindow dataset by crop_id (since it has no crop_name column).
    """
    crops_path = os.path.join(DATA_RAW_DIR, "crops.csv")
    if not os.path.exists(crops_path):
        return {}
    try:
        crops_df = pd.read_csv(crops_path)
    except Exception:
        return {}
    if "id" not in crops_df.columns or "standard_name" not in crops_df.columns:
        return {}
    out: dict[str, set[int]] = {}
    tmp = crops_df[["id", "standard_name"]].dropna(subset=["id"]).copy()
    for _, r in tmp.iterrows():
        std = str(r.get("standard_name") or "").strip()
        if not std:
            continue
        try:
            cid = int(float(r["id"]))
        except Exception:
            continue
        out.setdefault(std, set()).add(cid)
    return out


def load_dataset(ds_path: str, *, crop_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset CSV. Optionally filter rows by crop.

    crop_filter:
      - None: no filter (all crops)
      - "wheat_spring" / "sunflower": filters by crop_id sets from data_raw/crops.csv (standard_name)
      - numeric string like "43": filters by crop_id == 43
    """
    df = pd.read_csv(ds_path)
    if crop_filter is None:
        return df

    if "crop_id" not in df.columns:
        print("⚠️ crop_filter задан, но в датасете нет crop_id — фильтр не применён.")
        return df

    # numeric crop_id direct filter
    try:
        cid = int(float(crop_filter))
        return df[df["crop_id"].astype(float) == float(cid)].copy()
    except Exception:
        pass

    std2ids = _load_crops_standard_name_to_ids()
    ids = std2ids.get(str(crop_filter).strip())
    if not ids:
        print(f"⚠️ crop_filter='{crop_filter}' не найден в crops.csv.standard_name — фильтр не применён.")
        return df

    return df[df["crop_id"].astype(float).isin([float(x) for x in sorted(ids)])].copy()


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

LEAKY_FEATURES = [
    # Урожайность из операций уборки — почти всегда "post-harvest" сигнал (для прогноза заранее это leakage)
    "ops_yield_t_ha",
]


def prepare_features(
    df: pd.DataFrame,
    model_type: str = "all",
    *,
    exclude_leaky: bool = False,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, list]:
    """
    Подготовка фич и таргета.
    
    Args:
        df: Датасет
        model_type: 'no_ndvi', 'ndvi_only', 'all'
    
    Returns:
        X, y, groups (для GroupKFold), feature_cols
    """
    # Исключаем не-фичи
    exclude_cols = ['field_id', 'year', 'target_yield_t_ha', 'crop_name', 
                    'ops_season_start', 'ops_season_end']
    
    all_feature_cols = [col for col in df.columns if col not in exclude_cols]
    if exclude_leaky:
        all_feature_cols = [c for c in all_feature_cols if c not in LEAKY_FEATURES]
    
    # Разделяем фичи по категориям
    field_features = [c for c in all_feature_cols if c.startswith('field_')]
    crop_features = [c for c in all_feature_cols if c.startswith('crop_')]
    ops_features = [c for c in all_feature_cols if c.startswith('ops_')]
    ndvi_features = [c for c in all_feature_cols if c.startswith('ndvi_')]
    other_features = [c for c in all_feature_cols if c not in 
                     field_features + crop_features + ops_features + ndvi_features]
    
    # Выбираем фичи в зависимости от типа модели
    if model_type == 'no_ndvi':
        feature_cols = field_features + crop_features + ops_features + other_features
    elif model_type == 'ndvi_only':
        feature_cols = ndvi_features
    else:  # 'all'
        feature_cols = all_feature_cols
    
    X = df[feature_cols].copy()
    y = df["target_yield_t_ha"].copy()
    groups = df["year"].values  # Для GroupKFold

    # Оставляем только числовые фичи (строки типа ops_season_start уже исключены).
    # ВАЖНО: здесь не делаем заполнение пропусков — импутация выполняется внутри sklearn Pipeline в CV,
    # чтобы не было утечки статистик (медианы) из тестовых годов.
    X = X.select_dtypes(include=[np.number]).copy()

    # Drop features that are 100% NaN (otherwise imputers will warn/skip them).
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns ({len(all_nan_cols)}): {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    feature_cols = X.columns.tolist()

    return X, y, groups, feature_cols


def build_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Pipeline = (preprocess -> model)
    preprocess делает импутацию внутри CV (без leakage) и добавляет missing indicators.
    """
    # Отдельно имитируем categorical-like id'шники константой -1 (а не медианой).
    id_like = [c for c in ["crop_id", "prev_crop_id"] if c in feature_cols]
    other_num = [c for c in feature_cols if c not in id_like]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), other_num),
            ("id", SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True), id_like),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        loss="squared_error",
    )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_pipeline_with_params(
    feature_cols: list[str],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_samples_leaf: int,
    subsample: float,
) -> Pipeline:
    """
    Вариант build_pipeline с явной передачей гиперпараметров GBDT.
    Используется для перебора по сетке в Итерации 2 (Step A).
    """
    id_like = [c for c in ["crop_id", "prev_crop_id"] if c in feature_cols]
    other_num = [c for c in feature_cols if c not in id_like]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), other_num),
            ("id", SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True), id_like),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=RANDOM_STATE,
        loss="squared_error",
    )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_with_oof(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    cv: GroupKFold,
) -> dict:
    """
    Честная оценка: OOF predictions + fold metrics.
    Возвращает:
      - cv_*_mean/std
      - train_* (fit on full data)
      - oof_pred (out-of-fold predictions aligned with X)
      - fitted_pipeline (fit on full data)
      - feature_names_out (после preprocess)
    """
    y_arr = y.to_numpy(dtype=float)
    oof_pred = np.full(shape=(len(y_arr),), fill_value=np.nan, dtype=float)

    fold_r2: list[float] = []
    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    fold_mape: list[float] = []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Клонируем переданный pipeline, чтобы использовать одинаковые гиперпараметры
        # (как для baseline, так и для перебора по сетке гиперпараметров).
        pipeline_fold = clone(pipeline)
        pipeline_fold.fit(X_train, y_train)
        y_pred = pipeline_fold.predict(X_test)
        oof_pred[test_idx] = y_pred

        fold_r2.append(r2_score(y_test, y_pred))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
        fold_mae.append(float(mean_absolute_error(y_test, y_pred)))
        fold_mape.append(_mape(y_test, y_pred))

    # Train metrics (fit on all data)
    pipeline.fit(X, y_arr)
    y_train_pred = pipeline.predict(X)

    # Feature names after preprocessing
    try:
        feature_names_out = pipeline.named_steps["preprocess"].get_feature_names_out()
        feature_names_out = [str(x) for x in feature_names_out]
    except Exception:
        feature_names_out = [str(c) for c in X.columns]

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
        "oof_pred": oof_pred,
        "train_pred": y_train_pred,
        "fitted_pipeline": pipeline,
        "feature_names_out": feature_names_out,
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_type: str,
) -> dict:
    """Обучение + CV через OOF predictions (без leakage)."""
    print(f"🔧 Обучение модели: {model_type}")
    print(f"   Фич (raw): {X.shape[1]}")
    print()

    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    pipeline = build_pipeline(list(X.columns))

    print(f"  Cross-validation (GroupKFold по годам, {CV_FOLDS} folds) + OOF predictions...")
    res = evaluate_with_oof(pipeline, X, y, groups, group_kfold)

    print(f"  ✅ R² CV: {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")
    print(f"  ✅ RMSE CV: {res['cv_rmse_mean']:.4f} ± {res['cv_rmse_std']:.4f} т/га")
    print(f"  ✅ MAE CV: {res['cv_mae_mean']:.4f} ± {res['cv_mae_std']:.4f} т/га")
    print(f"  ✅ MAPE CV: {res['cv_mape_mean']:.2f}% ± {res['cv_mape_std']:.2f}%")
    print()

    print("📊 Метрики на полном датасете (train/in-sample):")
    print(f"  R²: {res['train_r2']:.4f}")
    print(f"  RMSE: {res['train_rmse']:.4f} т/га")
    print(f"  MAE: {res['train_mae']:.4f} т/га")
    print(f"  MAPE: {res['train_mape']:.2f}%")
    print()

    return {
        "pipeline": res["fitted_pipeline"],
        "model_type": model_type,
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
        "oof_pred": res["oof_pred"],
        "train_pred": res["train_pred"],
        "feature_names": res["feature_names_out"],
    }


def calculate_mape_cv(X, y, groups, cv, model_params):
    """
    Deprecated (Iteration 1):
    Раньше MAPE считался отдельно через обучение модели на каждом fold на "сырых" X.
    Сейчас всё считается через OOF predictions в evaluate_with_oof(), где импутация внутри Pipeline.
    """
    raise RuntimeError("calculate_mape_cv is deprecated; use train_model_with_pipeline() / evaluate_with_oof().")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(model_result: dict, dataset_name: str, top_n: int = 20):
    """
    Визуализация важности фич.
    """
    pipeline = model_result["pipeline"]
    feature_names = model_result["feature_names"]
    model_type = model_result['model_type']

    model = pipeline.named_steps["model"]
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"📊 Топ-{top_n} важных фич ({model_type}):")
    print(importance.head(top_n).to_string(index=False))
    print()
    
    # График
    plt.figure(figsize=(10, 8))
    top_features = importance.head(top_n)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance ({dataset_name} / {model_type})', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f'feature_importance_{dataset_name}_{model_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 График сохранен: {plot_path}")
    print()
    
    return importance


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, model_type: str, dataset_name: str):
    """
    Визуализация предсказаний vs реальные значения.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Yield (т/га)', fontsize=12)
    axes[0].set_ylabel('Predicted Yield (т/га)', fontsize=12)
    axes[0].set_title(f'Predictions vs Actual ({dataset_name} / {model_type})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Yield (т/га)', fontsize=12)
    axes[1].set_ylabel('Residuals (т/га)', fontsize=12)
    axes[1].set_title(f'Residuals Plot ({dataset_name})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f'predictions_plot_{dataset_name}_{model_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 График сохранен: {plot_path}")
    print()


def compare_models(models_results: list):
    """
    Сравнение метрик всех моделей.
    """
    comparison = []
    for result in models_results:
        model_type = result['model_type']
        if model_type.startswith("no_ndvi"):
            features_set = "no_ndvi"
        elif model_type.startswith("ndvi_only"):
            features_set = "ndvi_only"
        else:
            features_set = "all"

        comparison.append({
            'dataset_name': result.get('dataset_name', 'base_with_ndvi'),
            'features_set': features_set,
            'model_type': model_type,
            'n_features': len(result['feature_names']),
            'cv_r2_mean': result['cv_r2_mean'],
            'cv_r2_std': result['cv_r2_std'],
            'cv_rmse_mean': result['cv_rmse_mean'],
            'cv_rmse_std': result['cv_rmse_std'],
            'cv_mae_mean': result['cv_mae_mean'],
            'cv_mae_std': result['cv_mae_std'],
            'cv_mape_mean': result.get('cv_mape_mean', np.nan),
            'cv_mape_std': result.get('cv_mape_std', np.nan),
            'train_r2': result['train_r2'],
            'train_rmse': result['train_rmse'],
            'train_mae': result['train_mae'],
            'train_mape': result['train_mape'],
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("=" * 80)
    print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    print()
    print(comparison_df.to_string(index=False))
    print()
    
    # Save (append-only): do not erase previous rows
    comparison_path = os.path.join(OUTPUT_DIR, 'models_comparison.csv')
    if os.path.exists(comparison_path):
        try:
            prev = pd.read_csv(comparison_path)
            comparison_df = pd.concat([prev, comparison_df], ignore_index=True)
        except Exception:
            pass
    # Drop duplicates conservatively (same dataset/model_type/features_set)
    if {"dataset_name", "model_type", "features_set"}.issubset(comparison_df.columns):
        comparison_df = comparison_df.drop_duplicates(subset=["dataset_name", "model_type", "features_set"], keep="last")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"💾 Сравнение сохранено (append): {comparison_path}")
    print()
    
    return comparison_df


# ============================================================================
# MAIN
# ============================================================================

def train_on_dataset(df: pd.DataFrame, dataset_name: str) -> list:
    """
    Обучает 4 варианта моделей на одном датасете:
      - no_ndvi
      - no_ndvi_no_leak
      - ndvi_only
      - all
      - all_no_leak
    Возвращает список results с добавленным dataset_name.
    """
    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)
    print(f"Записей: {len(df)}, колонок: {len(df.columns)}")
    print()

    if "target_yield_t_ha" not in df.columns:
        print("❌ Таргет target_yield_t_ha не найден, пропускаю датасет")
        return []

    models_results: list = []

    # Валидация внутри train_model_with_pipeline (OOF)
    # Сервисные колонки для сохранения OOF-предсказаний
    service_cols = [c for c in ["field_id", "year", "crop_id", "target_yield_t_ha"] if c in df.columns]

    # 1) Без NDVI
    print("=" * 80)
    print(f"{dataset_name} — МОДЕЛЬ: no_ndvi (без NDVI)")
    print("=" * 80)
    X1, y1, g1, _ = prepare_features(df, model_type="no_ndvi", exclude_leaky=False)
    r1 = train_model_with_pipeline(X1, y1, g1, "no_ndvi")
    r1["dataset_name"] = dataset_name
    plot_feature_importance(r1, dataset_name, top_n=20)
    plot_predictions(y1, r1["oof_pred"], "no_ndvi_oof", dataset_name)
    if service_cols:
        oof_df = df[service_cols].copy()
        oof_df["model_type"] = "no_ndvi"
        oof_df["model_pred_oof_t_ha"] = r1["oof_pred"]
        oof_df["model_pred_train_t_ha"] = r1["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, f"oof_predictions_{dataset_name}_no_ndvi.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF predictions сохранены: {oof_path}")
    models_results.append(r1)

    # 1b) Без NDVI, без leakage
    print("=" * 80)
    print(f"{dataset_name} — МОДЕЛЬ: no_ndvi_no_leak (без NDVI, без ops_yield_t_ha)")
    print("=" * 80)
    X1b, y1b, g1b, _ = prepare_features(df, model_type="no_ndvi", exclude_leaky=True)
    r1b = train_model_with_pipeline(X1b, y1b, g1b, "no_ndvi_no_leak")
    r1b["dataset_name"] = dataset_name
    plot_feature_importance(r1b, dataset_name, top_n=20)
    plot_predictions(y1b, r1b["oof_pred"], "no_ndvi_no_leak_oof", dataset_name)
    if service_cols:
        oof_df = df[service_cols].copy()
        oof_df["model_type"] = "no_ndvi_no_leak"
        oof_df["model_pred_oof_t_ha"] = r1b["oof_pred"]
        oof_df["model_pred_train_t_ha"] = r1b["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, f"oof_predictions_{dataset_name}_no_ndvi_no_leak.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF predictions сохранены: {oof_path}")
    models_results.append(r1b)

    # 2) Только NDVI
    print("=" * 80)
    print(f"{dataset_name} — МОДЕЛЬ: ndvi_only")
    print("=" * 80)
    X2, y2, g2, f2 = prepare_features(df, model_type="ndvi_only", exclude_leaky=False)
    if len(f2) > 0:
        r2 = train_model_with_pipeline(X2, y2, g2, "ndvi_only")
        r2["dataset_name"] = dataset_name
        plot_feature_importance(r2, dataset_name, top_n=20)
        plot_predictions(y2, r2["oof_pred"], "ndvi_only_oof", dataset_name)
        if service_cols:
            oof_df = df[service_cols].copy()
            oof_df["model_type"] = "ndvi_only"
            oof_df["model_pred_oof_t_ha"] = r2["oof_pred"]
            oof_df["model_pred_train_t_ha"] = r2["train_pred"]
            oof_path = os.path.join(OUTPUT_DIR, f"oof_predictions_{dataset_name}_ndvi_only.csv")
            oof_df.to_csv(oof_path, index=False)
            print(f"💾 OOF predictions сохранены: {oof_path}")
        models_results.append(r2)
    else:
        print("⚠️ NDVI-фичи не найдены, пропускаю модель ndvi_only")
        print()

    # 3) Все признаки
    print("=" * 80)
    print(f"{dataset_name} — МОДЕЛЬ: all (все признаки)")
    print("=" * 80)
    X3, y3, g3, _ = prepare_features(df, model_type="all", exclude_leaky=False)
    r3 = train_model_with_pipeline(X3, y3, g3, "all")
    r3["dataset_name"] = dataset_name
    plot_feature_importance(r3, dataset_name, top_n=20)
    plot_predictions(y3, r3["oof_pred"], "all_oof", dataset_name)
    if service_cols:
        oof_df = df[service_cols].copy()
        oof_df["model_type"] = "all"
        oof_df["model_pred_oof_t_ha"] = r3["oof_pred"]
        oof_df["model_pred_train_t_ha"] = r3["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, f"oof_predictions_{dataset_name}_all.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF predictions сохранены: {oof_path}")
    models_results.append(r3)

    # 3b) Все признаки, без leakage
    print("=" * 80)
    print(f"{dataset_name} — МОДЕЛЬ: all_no_leak")
    print("=" * 80)
    X3b, y3b, g3b, _ = prepare_features(df, model_type="all", exclude_leaky=True)
    r3b = train_model_with_pipeline(X3b, y3b, g3b, "all_no_leak")
    r3b["dataset_name"] = dataset_name
    plot_feature_importance(r3b, dataset_name, top_n=20)
    plot_predictions(y3b, r3b["oof_pred"], "all_no_leak_oof", dataset_name)
    if service_cols:
        oof_df = df[service_cols].copy()
        oof_df["model_type"] = "all_no_leak"
        oof_df["model_pred_oof_t_ha"] = r3b["oof_pred"]
        oof_df["model_pred_train_t_ha"] = r3b["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, f"oof_predictions_{dataset_name}_all_no_leak.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF predictions сохранены: {oof_path}")
    models_results.append(r3b)

    return models_results


def main():
    print("=" * 80)
    print("BASELINE ML MODEL TRAINING - MULTI DATASETS")
    print("=" * 80)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results: list = []

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_path = ds["path"]
        print("=" * 80)
        print(f"📂 Загрузка датасета: {ds_name}")
        print(f"Файл: {ds_path}")
        print("=" * 80)
        if not os.path.exists(ds_path):
            print(f"⚠️ Файл не найден, пропускаю датасет {ds_name}")
            print()
            continue
        df = load_dataset(ds_path, crop_filter=None)
        results = train_on_dataset(df, ds_name)
        all_results.extend(results)

    if not all_results:
        print("❌ Нет результатов обучения — проверь наличие датасетов")
        return

    # Сравнение и сохранение (baseline конфигурации)
    comparison_df = compare_models(all_results)

    print("💾 Сохранение моделей...")
    for result in all_results:
        dataset_name = result.get("dataset_name", "base_with_ndvi")
        model_type = result["model_type"]
        model_path = os.path.join(OUTPUT_DIR, f"model_{dataset_name}_{model_type}.pkl")
        with open(model_path, "wb") as f:
            # Сохраняем sklearn Pipeline (preprocess + model)
            pickle.dump(result["pipeline"], f)
        print(f"  ✅ {model_path}")
    print()

    # Итоговый отчёт по лучшим моделям внутри каждого датасета
    print("=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЁТ ПО ДАТАСЕТАМ")
    print("=" * 80)
    print()
    for ds_name in sorted(set(r["dataset_name"] for r in all_results)):
        subset = [r for r in all_results if r["dataset_name"] == ds_name]
        best_r2 = max(subset, key=lambda x: x["cv_r2_mean"])
        best_rmse = min(subset, key=lambda x: x["cv_rmse_mean"])
        best_mape = min(subset, key=lambda x: x.get("cv_mape_mean", np.inf))
        print(f"Dataset: {ds_name}")
        print(f"  Best CV R²:   {best_r2['model_type']} = {best_r2['cv_r2_mean']:.4f}")
        print(f"  Best CV RMSE: {best_rmse['model_type']} = {best_rmse['cv_rmse_mean']:.4f} т/га")
        print(f"  Best CV MAPE: {best_mape['model_type']} = {best_mape.get('cv_mape_mean', np.nan):.2f}%")
        print()

    print("=" * 80)
    print("✅ ГОТОВО! (Baseline модели сохранены)")
    print("=" * 80)

    # ======================================================================
    # Итерация 2, шаг A: регуляризация GBDT на full_extended + all_no_leak
    # ======================================================================
    print()
    print("=" * 80)
    print("ITERATION 2 - STEP A: GBDT REGULARIZATION (full_extended / all_no_leak)")
    print("=" * 80)
    print()

    full_path = os.path.join(DATA_DIR, "ml_dataset_full_extended.csv")
    if not os.path.exists(full_path):
        print(f"⚠️ Файл {full_path} не найден, пропускаю grid search для full_extended.")
        return

    df_full = pd.read_csv(full_path)
    if "target_yield_t_ha" not in df_full.columns:
        print("❌ В full_extended нет target_yield_t_ha — пропускаю grid search.")
        return

    X_full, y_full, g_full, feature_cols_full = prepare_features(
        df_full, model_type="all", exclude_leaky=True
    )

    print(f"full_extended + all_no_leak: {X_full.shape[0]} наблюдений, {X_full.shape[1]} фич.")
    print()

    # Небольшая ручная сетка гиперпараметров (8 комбинаций)
    param_grid = [
        {"max_depth": 2, "min_samples_leaf": 5, "subsample": 0.7, "learning_rate": 0.1, "n_estimators": 150},
        {"max_depth": 2, "min_samples_leaf": 10, "subsample": 0.8, "learning_rate": 0.1, "n_estimators": 150},
        {"max_depth": 3, "min_samples_leaf": 5, "subsample": 0.7, "learning_rate": 0.1, "n_estimators": 200},
        {"max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8, "learning_rate": 0.1, "n_estimators": 200},
        {"max_depth": 2, "min_samples_leaf": 5, "subsample": 0.7, "learning_rate": 0.05, "n_estimators": 250},
        {"max_depth": 2, "min_samples_leaf": 10, "subsample": 0.8, "learning_rate": 0.05, "n_estimators": 250},
        {"max_depth": 3, "min_samples_leaf": 5, "subsample": 0.7, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8, "learning_rate": 0.05, "n_estimators": 300},
    ]

    grid_results = []
    cv = GroupKFold(n_splits=CV_FOLDS)

    for i, params in enumerate(param_grid, start=1):
        print("-" * 80)
        print(f"Config {i}/{len(param_grid)}: "
              f"depth={params['max_depth']}, "
              f"min_leaf={params['min_samples_leaf']}, "
              f"subsample={params['subsample']}, "
              f"lr={params['learning_rate']}, "
              f"n_estimators={params['n_estimators']}")

        pipeline_cfg = build_pipeline_with_params(
            feature_cols_full,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_samples_leaf=params["min_samples_leaf"],
            subsample=params["subsample"],
        )

        res_cfg = evaluate_with_oof(pipeline_cfg, X_full, y_full, g_full, cv)

        print(f"  -> CV R² = {res_cfg['cv_r2_mean']:.4f} ± {res_cfg['cv_r2_std']:.4f}")
        print(f"     CV RMSE = {res_cfg['cv_rmse_mean']:.4f} т/га")
        print(f"     CV MAPE = {res_cfg['cv_mape_mean']:.2f}%")
        print(f"     Train R² = {res_cfg['train_r2']:.4f}")
        print()

        row = {
            "dataset_name": "full_extended",
            "model_type": "all_no_leak_gbdt_grid",
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "subsample": params["subsample"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "cv_r2_mean": res_cfg["cv_r2_mean"],
            "cv_r2_std": res_cfg["cv_r2_std"],
            "cv_rmse_mean": res_cfg["cv_rmse_mean"],
            "cv_rmse_std": res_cfg["cv_rmse_std"],
            "cv_mae_mean": res_cfg["cv_mae_mean"],
            "cv_mae_std": res_cfg["cv_mae_std"],
            "cv_mape_mean": res_cfg["cv_mape_mean"],
            "cv_mape_std": res_cfg["cv_mape_std"],
            "train_r2": res_cfg["train_r2"],
            "train_rmse": res_cfg["train_rmse"],
            "train_mae": res_cfg["train_mae"],
            "train_mape": res_cfg["train_mape"],
        }
        grid_results.append(row)

    grid_df = pd.DataFrame(grid_results)
    grid_path = os.path.join(OUTPUT_DIR, "models_iteration2_gbdt_full_extended.csv")
    grid_df.to_csv(grid_path, index=False)
    print("-" * 80)
    print(f"💾 Результаты grid search сохранены: {grid_path}")

    best_row = grid_df.iloc[grid_df["cv_r2_mean"].idxmax()]
    print()
    print("🏆 Лучшая конфигурация (по CV R²) для full_extended + all_no_leak:")
    print(best_row.to_string())
    print()

    # ======================================================================
    # NEW: Crop-window dataset - train SAME best regularized GBDT in 3 modes
    # ======================================================================
    print("=" * 80)
    print("CROP-WINDOW v1: BEST REGULARIZED GBDT (all crops / wheat spring / sunflower)")
    print("=" * 80)
    print()

    cropwindow_path = os.path.join(DATA_DIR, "ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv")
    if not os.path.exists(cropwindow_path):
        print(f"⚠️ Файл {cropwindow_path} не найден, пропускаю crop-window обучение.")
        return

    def _train_one_cropwindow(tag: str, crop_filter: Optional[str]):
        df_cw = load_dataset(cropwindow_path, crop_filter=crop_filter)
        if df_cw.empty:
            print(f"⚠️ Пустой датасет после crop_filter={crop_filter} — пропускаю {tag}")
            return
        if "target_yield_t_ha" not in df_cw.columns:
            print(f"❌ Нет target_yield_t_ha в crop-window датасете — пропускаю {tag}")
            return

        X, y, g, feature_cols = prepare_features(df_cw, model_type="all", exclude_leaky=True)
        cv = GroupKFold(n_splits=CV_FOLDS)
        pipe = build_pipeline_with_params(
            feature_cols,
            n_estimators=BEST_REG_GBDT_PARAMS["n_estimators"],
            max_depth=BEST_REG_GBDT_PARAMS["max_depth"],
            learning_rate=BEST_REG_GBDT_PARAMS["learning_rate"],
            min_samples_leaf=BEST_REG_GBDT_PARAMS["min_samples_leaf"],
            subsample=BEST_REG_GBDT_PARAMS["subsample"],
        )
        res = evaluate_with_oof(pipe, X, y, g, cv)

        # Save OOF predictions
        oof_df = df_cw[[c for c in ["field_id", "year", "crop_id", "target_yield_t_ha"] if c in df_cw.columns]].copy()
        oof_df["model_pred_oof_t_ha"] = res["oof_pred"]
        oof_df["model_pred_train_t_ha"] = res["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, f"oof_full_ext_cropwindow_{tag}_all_no_leak_reg.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF сохранены: {oof_path}")

        # Plots (OOF)
        plot_predictions(y, res["oof_pred"], f"all_no_leak_reg_cropwindow_{tag}_oof", "full_extended_cropwindow_v1")

        # Append row to models_comparison.csv via compare_models()
        fake_result = {
            "dataset_name": f"full_extended_cropwindow_v1_{tag}",
            "model_type": f"all_no_leak_reg_cropwindow_{tag}",
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
            "feature_names": res["feature_names_out"],
        }
        compare_models([fake_result])

    _train_one_cropwindow("allcrops", None)
    _train_one_cropwindow("wheat", STANDARD_NAME_WHEAT_SPRING)
    _train_one_cropwindow("sunflower", STANDARD_NAME_SUNFLOWER)


if __name__ == "__main__":
    main()
