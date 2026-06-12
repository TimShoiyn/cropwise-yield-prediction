import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # подскажем юзеру в main


def load_data(
    fields_path: str = "data_raw/fields.csv",
    targets_path: str = "data_raw/targets.csv",
    field_id_col: str = "id",
    target_col: str = "target_ndvi",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Загружаем:
      - fields.csv из fetch_cropwise_data.py
      - targets.csv с таргетом продуктивности / NDVI по полям

    Ожидаемый формат targets.csv:
        field_id,target_ndvi
        195,0.72
        197,0.65
        ...
    """
    fields = pd.read_csv(fields_path)
    targets = pd.read_csv(targets_path)

    # нормализуем имена для join
    fields_id = field_id_col
    if fields_id not in fields.columns:
        raise ValueError(f"Колонка '{fields_id}' не найдена в {fields_path}")

    if "field_id" in targets.columns and field_id_col not in targets.columns:
        targets = targets.rename(columns={"field_id": field_id_col})

    if field_id_col not in targets.columns:
        raise ValueError(f"Колонка '{field_id_col}' не найдена в {targets_path}")
    if target_col not in targets.columns:
        raise ValueError(f"Колонка '{target_col}' не найдена в {targets_path}")

    df = pd.merge(
        fields,
        targets[[field_id_col, target_col]],
        on=field_id_col,
        how="inner",
    )

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def build_feature_matrix(
    X_raw: pd.DataFrame,
    field_id_col: str = "id",
    drop_geojson: bool = True,
    max_categories: int = 50,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Грубый feature engineering поверх fields.csv:
      - числовые фичи: площади, lat/long и т.п.
      - one-hot по region / admin / subadmin, но режем количество категорий.
    """
    df = X_raw.copy()

    # выкидываем идентификаторы и тяжёлую геометрию
    drop_cols: List[str] = [field_id_col, "external_id", "public_registry_key"]
    if drop_geojson and "shape_simplified_geojson" in df.columns:
        drop_cols.append("shape_simplified_geojson")
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # конвертим даты в numeric (timestamp)
    for col in ["created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9

    # базовые числовые колонки, которые наверняка есть
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    # категориальные: режем редкие категории
    cat_cols: List[str] = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= max_categories:
                cat_cols.append(col)

    # one-hot
    df_cat = pd.get_dummies(df[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=df.index)

    # собираем финальный X
    X_num = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, df_cat], axis=1)
    feature_names = list(X.columns)
    return X, feature_names


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Треним простую XGBoost-регрессию для прогноза таргета по полям.
    """
    if XGBRegressor is None:
        raise ImportError(
            "xgboost не установлен. Установи: pip install xgboost"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=os.cpu_count() or 4,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return model, (X_test, y_test, y_pred)


def main() -> None:
    """
    Сценарий:
      1) ждём, что у тебя уже есть:
           - data_raw/fields.csv (из fetch_cropwise_data.py)
           - data_raw/targets.csv с колонками: field_id,target_ndvi
      2) собираем фичи
      3) треним XGBoost
    """
    X_raw, y = load_data(
        fields_path="data_raw/fields.csv",
        targets_path="data_raw/targets.csv",
        field_id_col="id",
        target_col="target_ndvi",
    )
    X, feature_names = build_feature_matrix(X_raw, field_id_col="id")
    print(f"Фич: {len(feature_names)}, объектов: {len(X)}")

    model, _ = train_xgboost(X, y)

    # можно сохранить модель, если нужно
    # import joblib
    # joblib.dump(model, "models/xgb_ndvi.pkl")


if __name__ == "__main__":
    main()

