"""
Iteration 2 - Best GBDT model on full_extended + all_no_leak.

Цель:
    - Получить OOF-предсказания для лучшей регуляризованной модели GBDT:
        depth=3, min_samples_leaf=5, subsample=0.7, learning_rate=0.05, n_estimators=300
    - Сохранить:
        models/oof_predictions_full_extended_all_no_leak_reg.csv
        models/feature_importance_full_extended_all_no_leak_reg.png
        models/predictions_plot_full_extended_all_no_leak_reg_oof.png

Запуск:
    python train_iteration2_best_gbdt_full_extended.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train_baseline_model import (
    DATA_DIR,
    OUTPUT_DIR,
    CV_FOLDS,
    RANDOM_STATE,
    prepare_features,
    build_pipeline_with_params,
    evaluate_with_oof,
    plot_feature_importance,
    plot_predictions,
)


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


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

    print("=== Iteration 2: Best GBDT on full_extended + all_no_leak ===")
    print(f"Файл: {full_path}")
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")
    print()

    # Те же фичи, что и для all_no_leak (exclude_leaky=True)
    X, y, groups, feature_cols = prepare_features(df, model_type="all", exclude_leaky=True)
    print(f"Фич (raw): {X.shape[1]}")
    print()

    # Лучшие гиперпараметры из models_iteration2_gbdt_full_extended.csv
    best_params = {
        "max_depth": 3,
        "min_samples_leaf": 5,
        "subsample": 0.7,
        "learning_rate": 0.05,
        "n_estimators": 300,
    }

    print(
        "Используем лучшие гиперпараметры GBDT: "
        f"depth={best_params['max_depth']}, "
        f"min_leaf={best_params['min_samples_leaf']}, "
        f"subsample={best_params['subsample']}, "
        f"lr={best_params['learning_rate']}, "
        f"n_estimators={best_params['n_estimators']}"
    )
    print()

    pipeline = build_pipeline_with_params(
        feature_cols,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        min_samples_leaf=best_params["min_samples_leaf"],
        subsample=best_params["subsample"],
    )

    cv = GroupKFold(n_splits=CV_FOLDS)
    res = evaluate_with_oof(pipeline, X, y, groups, cv)

    print(
        f"CV R² = {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}, "
        f"CV RMSE = {res['cv_rmse_mean']:.4f} т/га, "
        f"CV MAPE = {res['cv_mape_mean']:.2f}%, "
        f"Train R² = {res['train_r2']:.4f}"
    )
    print()

    # Собираем результат в формате, совместимом с plot_* и downstream-скриптами
    model_result = {
        "pipeline": res["fitted_pipeline"],
        "model_type": "all_no_leak_reg",
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

    dataset_name = "full_extended"

    # 1) Feature importance
    plot_feature_importance(model_result, dataset_name, top_n=20)

    # 2) Predictions vs actual (OOF)
    plot_predictions(
        y_true=y,
        y_pred=res["oof_pred"],
        model_type="all_no_leak_reg_oof",
        dataset_name=dataset_name,
    )

    # 3) OOF-предсказания на диск (для compare_by_crops_and_cropwise.py и анализа по культурам)
    service_cols = [c for c in ["field_id", "year", "crop_id", "target_yield_t_ha"] if c in df.columns]
    if service_cols:
        oof_df = df[service_cols].copy()
        oof_df["model_type"] = "all_no_leak_reg"
        oof_df["model_pred_oof_t_ha"] = res["oof_pred"]
        oof_df["model_pred_train_t_ha"] = res["train_pred"]
        oof_path = os.path.join(OUTPUT_DIR, "oof_predictions_full_extended_all_no_leak_reg.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"💾 OOF predictions сохранены: {oof_path}")
        print()


if __name__ == "__main__":
    main()

