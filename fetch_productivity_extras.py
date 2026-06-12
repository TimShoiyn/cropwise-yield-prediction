"""
Выгрузка дополнительных данных по урожайности:
- /api/v3/productivity_estimate_histories
- /api/v3/productivity_estimate_peers

Цель: сохранить всё «на диск», чтобы потом при желании прикрутить к ML‑датасету.

Авторизация:
- использует тот же токен, что и остальные скрипты:
    $env:CROPWISE_API_KEY = "...."
"""

import os
from typing import Any, Dict, List

import pandas as pd

from fetch_cropwise_data import CropwiseClient, API_KEY, BASE_URL


DATA_DIR = "data_raw"
os.makedirs(DATA_DIR, exist_ok=True)

PROD_EST_HIST_CSV = os.path.join(DATA_DIR, "productivity_estimate_histories.csv")
PROD_EST_PEERS_CSV = os.path.join(DATA_DIR, "productivity_estimate_peers.csv")


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


def fetch_productivity_estimate_histories() -> pd.DataFrame:
    """
    GET /api/v3/productivity_estimate_histories
    Сохраняем estimate_history как JSON-строку (date→estimate_value).
    """
    print("=" * 80)
    print("ВЫГРУЗКА PRODUCTIVITY_ESTIMATE_HISTORIES")
    print("=" * 80)

    _ensure_api_key()
    client = CropwiseClient(API_KEY, base_url=BASE_URL)

    all_items: List[Dict[str, Any]] = []
    from_id: int | None = None
    page = 1

    while True:
        params: Dict[str, Any] = {"limit": 1000}
        if from_id is not None:
            params["from_id"] = from_id

        print(f"  -> Страница {page}, from_id={from_id} ...")
        resp = client.get("/productivity_estimate_histories", params=params)
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
            print("  ℹ️ meta.response.last_record_id не найден — останавливаемся")
            break

        from_id = int(last_id) + 1
        page += 1

    if not all_items:
        print("⚠️ Не получили ни одной записи productivity_estimate_histories")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    print(f"\n✅ Получено ProductivityEstimateHistory: {len(df)} записей")
    print("Колонки:", list(df.columns))

    # estimate_history — JSON с историей прогнозов по датам; сохраняем как есть
    df.to_csv(PROD_EST_HIST_CSV, index=False)
    print(f"💾 Сохранено: {PROD_EST_HIST_CSV}")
    return df


def fetch_productivity_estimate_peers() -> pd.DataFrame:
    """
    GET /api/v3/productivity_estimate_peers
    Для каждого productivity_estimate — «peer»-поля с их урожайностью, осадками, NDVI и т.д.
    """
    print("\n" + "=" * 80)
    print("ВЫГРУЗКА PRODUCTIVITY_ESTIMATE_PEERS")
    print("=" * 80)

    _ensure_api_key()
    client = CropwiseClient(API_KEY, base_url=BASE_URL)

    all_items: List[Dict[str, Any]] = []
    from_id: int | None = None
    page = 1

    while True:
        params: Dict[str, Any] = {"limit": 1000}
        if from_id is not None:
            params["from_id"] = from_id

        print(f"  -> Страница {page}, from_id={from_id} ...")
        resp = client.get("/productivity_estimate_peers", params=params)
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
            print("  ℹ️ meta.response.last_record_id не найден — останавливаемся")
            break

        from_id = int(last_id) + 1
        page += 1

    if not all_items:
        print("⚠️ Не получили ни одной записи productivity_estimate_peers")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    print(f"\n✅ Получено ProductivityEstimatePeers: {len(df)} записей")
    print("Колонки:", list(df.columns))

    df.to_csv(PROD_EST_PEERS_CSV, index=False)
    print(f"💾 Сохранено: {PROD_EST_PEERS_CSV}")
    return df


def main() -> None:
    fetch_productivity_estimate_histories()
    fetch_productivity_estimate_peers()
    print("\n" + "=" * 80)
    print("✅ ВЫГРУЗКА ДОП. ДАННЫХ ПО УРОЖАЙНОСТИ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

