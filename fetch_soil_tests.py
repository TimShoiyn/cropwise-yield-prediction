"""
Выгрузка почвенных анализов (SoilTest, SoilTestSample) из Cropwise Operations API.

Что делает:
- GET /api/v3a/soil_tests → сохраняет в data_raw/soil_tests.csv c распакованным elements.
- GET /api/v3/soil_test_samples → сохраняет в data_raw/soil_test_samples.csv c распакованным elements и координатами.

Авторизация:
- Использует тот же токен, что и остальной проект:
    $env:CROPWISE_API_KEY = "...."
"""

import os
from typing import Any, Dict, List

import pandas as pd

from fetch_cropwise_data import CropwiseClient, API_KEY, BASE_URL


DATA_DIR = "data_raw"
os.makedirs(DATA_DIR, exist_ok=True)

SOIL_TESTS_CSV = os.path.join(DATA_DIR, "soil_tests.csv")
SOIL_TEST_SAMPLES_CSV = os.path.join(DATA_DIR, "soil_test_samples.csv")


def _ensure_api_key() -> None:
    if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError(
            "❌ CROPWISE_API_KEY не задан. "
            "В PowerShell:  $env:CROPWISE_API_KEY = '...токен...'"
        )


def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        if "items" in obj and isinstance(obj["items"], list):
            return obj["items"]
        return [obj]
    return [obj]


def fetch_soil_tests() -> pd.DataFrame:
    """
    Выгружает все SoilTest через /api/v3a/soil_tests с простейшей пагинацией по from_id.
    """
    print("=" * 80)
    print("ВЫГРУЗКА SOIL TESTS (POЧВА)")
    print("=" * 80)

    _ensure_api_key()

    client = CropwiseClient(
        API_KEY,
        base_url="https://operations.cropwise.com/api/v3a/",
    )

    all_items: List[Dict[str, Any]] = []
    from_id: int | None = None
    page = 1

    while True:
        params: Dict[str, Any] = {"limit": 1000}
        if from_id is not None:
            params["from_id"] = from_id

        print(f"  -> Страница {page}, from_id={from_id} ...")
        resp = client.get("/soil_tests", params=params)
        items = _as_list(resp)
        if not items:
            print("  ✅ Достигнут конец данных (пустой список)")
            break

        all_items.extend(items)
        # Пытаемся вытащить last_record_id из meta, как в history_items
        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        last_id = None
        if isinstance(meta, dict):
            resp_meta = meta.get("response", {})
            if isinstance(resp_meta, dict):
                last_id = resp_meta.get("last_record_id")

        if not last_id:
            print("  ℹ️ meta.response.last_record_id не найден — считаем, что данных больше нет")
            break

        from_id = int(last_id) + 1
        page += 1

    if not all_items:
        print("⚠️ SoilTests не найдены (пустой ответ)")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    print(f"\n✅ Получено SoilTests: {len(df)} записей")
    print("Колонки:", list(df.columns))

    # elements: dict с pH, N, P, K, organic_matter, soil_organic_carbon и т.д.
    if "elements" in df.columns:
        elements_df = pd.json_normalize(df["elements"]).add_prefix("soil_")
        df = pd.concat([df.drop(columns=["elements"]), elements_df], axis=1)

    df.to_csv(SOIL_TESTS_CSV, index=False)
    print(f"💾 Сохранено: {SOIL_TESTS_CSV}")
    return df


def fetch_soil_test_samples() -> pd.DataFrame:
    """
    Выгружает SoilTestSample через /api/v3/soil_test_samples и распаковывает elements.
    """
    print("\n" + "=" * 80)
    print("ВЫГРУЗКА SOIL TEST SAMPLES (ПРОБЫ ПОЧВЫ)")
    print("=" * 80)

    _ensure_api_key()

    client = CropwiseClient(
        API_KEY,
        base_url=BASE_URL,  # v3
    )

    all_items: List[Dict[str, Any]] = []
    from_id: int | None = None
    page = 1

    while True:
        params: Dict[str, Any] = {"limit": 1000}
        if from_id is not None:
            params["from_id"] = from_id

        print(f"  -> Страница {page}, from_id={from_id} ...")
        resp = client.get("/soil_test_samples", params=params)
        items = _as_list(resp)
        if not items:
            print("  ✅ Достигнут конец данных (пустой список)")
            break

        all_items.extend(items)
        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        last_id = None
        if isinstance(meta, dict):
            resp_meta = meta.get("response", {})
            if isinstance(resp_meta, dict):
                last_id = resp_meta.get("last_record_id")

        if not last_id:
            print("  ℹ️ meta.response.last_record_id не найден — считаем, что данных больше нет")
            break

        from_id = int(last_id) + 1
        page += 1

    if not all_items:
        print("⚠️ SoilTestSamples не найдены (пустой ответ)")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    print(f"\n✅ Получено SoilTestSamples: {len(df)} записей")
    print("Колонки:", list(df.columns))

    # elements → soil_* фичи
    if "elements" in df.columns:
        elements_df = pd.json_normalize(df["elements"]).add_prefix("soil_")
        df = pd.concat([df.drop(columns=["elements"]), elements_df], axis=1)

    df.to_csv(SOIL_TEST_SAMPLES_CSV, index=False)
    print(f"💾 Сохранено: {SOIL_TEST_SAMPLES_CSV}")
    return df


def main() -> None:
    fetch_soil_tests()
    fetch_soil_test_samples()
    print("\n" + "=" * 80)
    print("✅ ВЫГРУЗКА ПОЧВЫ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

