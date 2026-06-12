"""
Временной split: train на 2010–2022, test на 2023–2025.
Более реалистичная оценка для production (прогноз будущих лет).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_PATH = ROOT_DIR / "data_processed" / "ml_dataset_with_crops_and_ndvi_v2.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"❌ Файл не найден: {DATA_PATH}. Сначала запусти build_ml_dataset.py")

    df = pd.read_csv(DATA_PATH)

    # Temporal split
    train_df = df[df["year"] <= 2022].copy()
    test_df = df[df["year"] >= 2023].copy()

    print(f"Train: {len(train_df)} строк (годы ≤2022)")
    print(f"Test:  {len(test_df)} строк (годы ≥2023)")

    # Таргет
    y_train = train_df["target_yield_t_ha"]
    y_test = test_df["target_yield_t_ha"]

    # Топ-20 фич + year dummies
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

    year_cols = [c for c in df.columns if c.startswith("year_")]
    features = [f for f in (top_features + year_cols) if f in df.columns]

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    # Импутация по train-части
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Модель
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Метрики
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n📊 РЕЗУЛЬТАТЫ (Temporal Split 2010–2022 → 2023–2025):")
    print(f"  Train R²: {r2_train:.4f}, RMSE: {rmse_train:.2f} т/га")
    print(f"  Test  R²: {r2_test:.4f}, RMSE: {rmse_test:.2f} т/га")
    print(f"  Overfit gap: {r2_train - r2_test:.4f}")


if __name__ == "__main__":
    main()

