"""
Выгрузить точный NPK-состав удобрений из /fertilizers.
Цель: заменить упрощённый маппинг 30/10/10 на реальный NPK.
"""

import os

import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

API_TOKEN = os.getenv("CROPWISE_API_KEY", "your_token_here")  # <-- УСТАНОВИ CROPWISE_API_KEY
BASE_URL = "https://operations.cropwise.com/api/v3"

FERTILIZERS_CSV = "data_raw/fertilizers_npk.csv"


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "X-User-Api-Token": API_TOKEN,
            "Accept": "application/json",
        }
    )
    return s


def main() -> None:
    print("=" * 80)
    print("ВЫГРУЗКА УДОБРЕНИЙ С NPK СОСТАВОМ (`/fertilizers`)")
    print("=" * 80)

    if API_TOKEN == "your_token_here":
        print("\n❌ ОШИБКА: Не указан API токен (CROPWISE_API_KEY)!")
        return

    session = get_session()
    url = f"{BASE_URL}/fertilizers"
    resp = session.get(url, timeout=40)

    if resp.status_code != 200:
        print(f"\n❌ Ошибка {resp.status_code}: {resp.text[:200]}")
        return

    data = resp.json().get("data", [])
    fert_df = pd.DataFrame(data)
    print(f"\n✅ Получено удобрений: {len(fert_df)}")
    print(f"  Колонки: {list(fert_df.columns)}")

    if "elements" in fert_df.columns:
        elements_df = pd.json_normalize(fert_df["elements"])
        base_cols = [
            "id",
            "name",
            "manufacturer_name",
            "fertilizer_type",
            "source_type",
            "nutrient_type",
        ]
        base_cols = [c for c in base_cols if c in fert_df.columns]
        fert_with_npk = pd.concat([fert_df[base_cols], elements_df], axis=1)
        fert_with_npk.to_csv(FERTILIZERS_CSV, index=False)
        print(f"\nNPK/элементы колонки: {list(elements_df.columns)}")
        print(f"💾 Сохранено: {FERTILIZERS_CSV}")
        if {"N", "P2O5", "K2O"}.issubset(fert_with_npk.columns):
            print("\nПримеры удобрений с NPK:")
            print(fert_with_npk[["id", "name", "N", "P2O5", "K2O"]].head(10))
    else:
        fert_df.to_csv(FERTILIZERS_CSV, index=False)
        print("\n⚠️ Колонка `elements` не найдена, сохранили как есть.")
        print(f"💾 Сохранено: {FERTILIZERS_CSV}")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()

