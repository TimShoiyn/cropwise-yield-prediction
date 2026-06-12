"""
Оптимизированная модель: топ-20 фич + year + упрощённая архитектура.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_PATH = ROOT_DIR / "data_processed" / "ml_dataset_with_crops_and_ndvi_v2.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"❌ Файл не найден: {DATA_PATH}. Сначала запусти build_ml_dataset.py")

    df = pd.read_csv(DATA_PATH)

    # Таргет
    y = df["target_yield_t_ha"]

    # Топ-20 фич (по важности из all_no_leak) + позже добавим year dummies
    top_features = [
        "ops_count_total",
        "ndvi_max_season",
        "ndvi_trend",
        "ndvi_std_season",
        "ndvi_early",
        "ndvi_peak_doy",
        "ndvi_mid",
        "crop_id",
        "ndvi_integral",
        "ndvi_mean_season",
        "field_lat",
        "ndvi_late",
        "ndvi_min_season",
        "ndvi_observations",
        "field_long",
        "field_tillable_area",
        "ndvi_green_period_days",
        "ops_count_soil",
        "field_calculated_area",
        "ops_days_seeding_to_ndvi_peak",
    ]

    # year dummies, которые мы добавили в build_ml_dataset.py
    year_cols = [c for c in df.columns if c.startswith("year_")]
    top_features_with_year = top_features + year_cols

    # Фичи, которые реально есть в датасете
    available_features = [f for f in top_features_with_year if f in df.columns]
    print(f"Используемые фичи: {len(available_features)}")
    print("  ", available_features)

    X = df[available_features].copy()

    # Заполнить пропуски медианой по train-части (тут весь датасет используется как train)
    X = X.fillna(X.median(numeric_only=True))

    # Модель с уменьшенной сложностью (меньше переобучения)
    model = GradientBoostingRegressor(
        n_estimators=50,  # Было 100 → уменьшили
        max_depth=3,  # Было 5 → уменьшили
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation по годам
    cv = GroupKFold(n_splits=5)
    groups = df["year"]

    scores_r2 = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2")
    scores_rmse = -cross_val_score(
        model, X, y, cv=cv, groups=groups, scoring="neg_root_mean_squared_error"
    )

    print("\n📊 РЕЗУЛЬТАТЫ (топ-20 фич + year, упрощённая модель):")
    print(f"  CV R²:   {scores_r2.mean():.4f} ± {scores_r2.std():.44f}")
    print(f"  CV RMSE: {scores_rmse.mean():.4f} ± {scores_rmse.std():.4f} т/га")

    # Обучить на полном датасете
    model.fit(X, y)
    y_pred_train = model.predict(X)
    r2_train = 1 - ((y - y_pred_train) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse_train = float(np.sqrt(((y - y_pred_train) ** 2).mean()))

    print(f"\n  Train R²:   {r2_train:.4f}")
    print(f"  Train RMSE: {rmse_train:.4f} т/га")
    print(f"  Train/CV gap: {r2_train - scores_r2.mean():.4f}")


if __name__ == "__main__":
    main()

