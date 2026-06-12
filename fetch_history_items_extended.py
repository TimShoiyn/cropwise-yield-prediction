"""
Выгрузить history_items и добрать из них доп. урожайность.
Цель: найти field×year с урожайностью, которых нет в productivity_estimates.
"""

import os
import time
from typing import Dict, Any, List

import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

API_TOKEN = os.getenv("CROPWISE_API_KEY", "your_token_here")  # <-- УСТАНОВИ CROPWISE_API_KEY
BASE_URL = "https://operations.cropwise.com/api/v3"

FIELDS_CSV = "data_raw/fields.csv"
PRODUCTIVITY_CSV = "data_raw/productivity_estimates.csv"

HISTORY_ITEMS_CSV = "data_raw/history_items_full.csv"
ADDITIONAL_YIELDS_CSV = "data_raw/additional_yields_from_history.csv"


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "X-User-Api-Token": API_TOKEN,
            "Accept": "application/json",
        }
    )
    return s


def fetch_all_history_items(session: requests.Session, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Пагинация по /history_items через from_id/limit.
    По докам Cropwise: meta.response.last_record_id.
    """
    print("\n1) ВЫГРУЗКА HISTORY ITEMS С ПАГИНАЦИЕЙ (`/history_items`)")
    url = f"{BASE_URL}/history_items"

    all_items: List[Dict[str, Any]] = []
    from_id = 0
    page = 1

    while True:
        print(f"  Страница {page} (from_id={from_id})...")
        params = {"from_id": from_id, "limit": limit}
        resp = session.get(url, params=params, timeout=40)

        if resp.status_code != 200:
            print(f"  ❌ Ошибка {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        items = data.get("data", [])
        if not items:
            print("  ✅ Достигнут конец данных")
            break

        all_items.extend(items)
        print(f"    Получено: {len(items)} записей, всего: {len(all_items)}")

        meta = data.get("meta", {}).get("response", {})
        last_id = meta.get("last_record_id")
        if last_id is None:
            print("  ⚠️ last_record_id не найден в meta — останавливаемся")
            break

        from_id = last_id + 1
        page += 1
        time.sleep(0.3)

    return all_items


def extract_additional_yields(history_df: pd.DataFrame, productivity_df: pd.DataFrame) -> pd.DataFrame:
    print("\n2) ПОИСК ДОПОЛНИТЕЛЬНОЙ УРОЖАЙНОСТИ В history_items")

    if history_df.empty:
        print("  ⚠️ history_df пустой — нечего искать")
        return pd.DataFrame()

    required_cols = {"field_id", "year", "productivity"}
    if not required_cols.issubset(history_df.columns):
        print(f"  ⚠️ В history_items нет нужных колонок {required_cols}")
        return pd.DataFrame()

    existing_field_years = set(zip(productivity_df["field_id"], productivity_df["year"]))
    print(f"  Существующих field×year в productivity_estimates: {len(existing_field_years)}")

    history_with_yield = history_df[history_df["productivity"].notna()].copy()
    print(f"  History items с ненулевой урожайностью: {len(history_with_yield)}")

    history_with_yield["field_year"] = list(
        zip(history_with_yield["field_id"], history_with_yield["year"])
    )
    mask_new = ~history_with_yield["field_year"].isin(existing_field_years)
    new_yields = history_with_yield[mask_new]
    print(f"  НОВЫХ field×year с урожайностью: {len(new_yields)}")

    if new_yields.empty:
        return pd.DataFrame()

    cols = [
        "field_id",
        "year",
        "productivity",
        "crop_id",
        "variety",
        "sowing_date",
        "harvesting_date",
    ]
    cols = [c for c in cols if c in new_yields.columns]

    additional = new_yields[cols].rename(columns={"productivity": "target_yield_t_ha"})
    return additional


def main() -> None:
    print("=" * 80)
    print("ВЫГРУЗКА HISTORY ITEMS И ДОП. УРОЖАЙНОСТИ")
    print("=" * 80)

    if API_TOKEN == "your_token_here":
        print("\n❌ ОШИБКА: Не указан API токен (CROPWISE_API_KEY)!")
        return

    session = get_session()

    # 1) Забираем все history_items
    all_items = fetch_all_history_items(session, limit=1000)
    history_df = pd.DataFrame(all_items)
    print(f"\n✅ Всего получено {len(history_df)} history_items")
    if not history_df.empty:
        print(f"  Колонки: {list(history_df.columns)}")
        history_df.to_csv(HISTORY_ITEMS_CSV, index=False)
        print(f"  💾 Сохранено: {HISTORY_ITEMS_CSV}")

    # 2) Добираем урожайность
    if not os.path.exists(PRODUCTIVITY_CSV):
        print(f"\n⚠️ Файл productivity_estimates не найден: {PRODUCTIVITY_CSV}")
        return

    productivity_df = pd.read_csv(PRODUCTIVITY_CSV)
    if productivity_df.empty:
        print("\n⚠️ productivity_estimates пустой — расширять нечего")
        return

    additional_yields = extract_additional_yields(history_df, productivity_df)
    if additional_yields.empty:
        print("  ℹ️ Новой урожайности в history_items не нашли")
    else:
        additional_yields.to_csv(ADDITIONAL_YIELDS_CSV, index=False)
        print(f"  💾 Доп. урожайность сохранена в {ADDITIONAL_YIELDS_CSV}")
        print("\n  Примеры:")
        print(additional_yields.head(10))

    # 3) Немного статистики
    if not history_df.empty:
        print("\n3) СТАТИСТИКА ПО HISTORY ITEMS")
        if "year" in history_df.columns:
            print("\n  Распределение по годам:")
            print(history_df["year"].value_counts().sort_index())
        if "crop_id" in history_df.columns:
            print("\n  Топ культур (crop_id):")
            print(history_df["crop_id"].value_counts().head(10))
        if {"field_id", "year"}.issubset(history_df.columns):
            hist_fy = (
                history_df.groupby(["field_id", "year"])
                .size()
                .reset_index(name="count")
            )
            print(f"\n  Уникальных field×year в history_items: {len(hist_fy)}")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()

