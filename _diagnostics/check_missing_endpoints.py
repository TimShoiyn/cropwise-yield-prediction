"""
Проверка недостающих эндпоинтов из Cropwise Operations API.
Документация: https://cropwiseoperations.docs.apiary.io/
"""

import json
import os
import time
from typing import Dict, Any

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

# Токен берём из ENV, чтобы не светить его в коде
API_TOKEN = os.getenv("CROPWISE_API_KEY", "your_token_here")  # <-- УСТАНОВИ CROPWISE_API_KEY
BASE_URL = "https://operations.cropwise.com/api/v3"

# Поле по умолчанию для эндпоинтов вида /fields/{id}/...
TEST_FIELD_ID = 195

# ============================================================================
# ЭНДПОИНТЫ, КОТОРЫЕ МЫ НЕ ПРОВЕРИЛИ
# ============================================================================

ENDPOINTS_TO_CHECK = [
    # 1. Weather / Метео (КРИТИЧНО!)
    {
        "name": "Virtual Weather Stations",
        "endpoint": "/virtual_weather_stations",
        "description": "Метеостанции для полей - температура, осадки",
    },
    {
        "name": "Weather History",
        "endpoint": "/weather_history",
        "description": "История погоды по станциям",
    },
    # 2. Soil / Почва
    {
        "name": "Soil Samples",
        "endpoint": "/soil_samples",
        "description": "Пробы почвы - pH, NPK, органика",
    },
    {
        "name": "Soil Maps",
        "endpoint": "/soil_maps",
        "description": "Карты почв по полям",
    },
    # 3. History Items / История
    {
        "name": "History Items",
        "endpoint": "/history_items",
        "description": "Полная история полей - культуры, урожайность, операции",
    },
    # 4. Crops on Fields / Культуры на полях
    {
        "name": "Field History",
        "endpoint": "/fields/{id}/history",
        "description": "История конкретного поля по годам",
    },
    # 5. Machine Work / Работа техники
    {
        "name": "Machine Work Results",
        "endpoint": "/machine_work_results",
        "description": "Результаты работы техники - GPS треки, расход топлива",
    },
    # 6. Application Mix / Что вносили
    {
        "name": "Fertilizers",
        "endpoint": "/fertilizers",
        "description": "Справочник удобрений с NPK составом",
    },
    {
        "name": "Pesticides",
        "endpoint": "/pesticides",
        "description": "Справочник СЗР",
    },
    # 7. Organization / Организация
    {
        "name": "Organization",
        "endpoint": "/organization",
        "description": "Информация об организации, контракты",
    },
    # 8. Seasons / Сезоны
    {
        "name": "Seasons",
        "endpoint": "/seasons",
        "description": "Список сезонов с датами начала/конца",
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_url(endpoint_template: str, field_id: int) -> str:
    """
    Подставляем field_id в {id}, если есть, и собираем абсолютный URL.
    """
    endpoint = endpoint_template.replace("{id}", str(field_id))
    return f"{BASE_URL}{endpoint}"


def check_endpoint(endpoint_info: Dict[str, Any], token: str, field_id: int) -> Dict[str, Any]:
    """
    Проверить доступность эндпоинта.
    """
    url = build_url(endpoint_info["endpoint"], field_id)

    # В Cropwise Operations используется X-User-Api-Token, не Bearer
    headers = {
        "X-User-Api-Token": token,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        status = resp.status_code

        if status == 200:
            try:
                data = resp.json()
            except Exception:
                data = None

            count = None
            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    count = len(data["data"])
                elif isinstance(data.get("items"), list):
                    count = len(data["items"])
                elif isinstance(data.get("results"), list):
                    count = len(data["results"])
                else:
                    count = 1
            elif isinstance(data, list):
                count = len(data)

            sample_str = ""
            if data is not None:
                try:
                    sample_str = json.dumps(data, ensure_ascii=False)[:500]
                except Exception:
                    sample_str = str(data)[:500]

            return {
                "available": True,
                "status_code": status,
                "count": count,
                "sample": sample_str,
                "url": url,
            }

        # 4xx/5xx — просто логируем
        return {
            "available": False,
            "status_code": status,
            "error": resp.text[:300],
            "url": url,
        }

    except Exception as e:
        return {
            "available": False,
            "status_code": None,
            "error": str(e)[:300],
            "url": url,
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ПРОВЕРКА НЕДОСТАЮЩИХ ЭНДПОИНТОВ CROPWISE API")
    print("=" * 80)

    if API_TOKEN == "your_token_here":
        print("\n❌ ОШИБКА: Не указан API токен!")
        print("   Установи переменную окружения CROPWISE_API_KEY перед запуском.")
        print('   Пример (PowerShell):  $env:CROPWISE_API_KEY = "ТОКЕН"')
        raise SystemExit(1)

    print(f"\n✅ Токен настроен (длина: {len(API_TOKEN)} символов)")
    print(f"   Проверяем {len(ENDPOINTS_TO_CHECK)} эндпоинтов...")

    results: list[Dict[str, Any]] = []

    for i, endpoint_info in enumerate(ENDPOINTS_TO_CHECK, 1):
        print(f"\n[{i}/{len(ENDPOINTS_TO_CHECK)}] {endpoint_info['name']}")
        print(f"    Endpoint: {endpoint_info['endpoint']}")
        print(f"    Описание: {endpoint_info['description']}")
        print(f"    Проверяем...", end=" ")

        res = check_endpoint(endpoint_info, API_TOKEN, TEST_FIELD_ID)

        if res["available"]:
            print(f"✅ ДОСТУПЕН! (Status: {res['status_code']})")
            if res.get("count") is not None:
                print(f"    📊 Записей (примерная оценка): {res['count']}")
            if res.get("sample"):
                print("    Пример данных:")
                print(f"    {res['sample']}...")
        else:
            print(f"❌ Недоступен (Status: {res['status_code']}, Error: {res.get('error', 'Unknown')})")

        results.append({**endpoint_info, **res})

        time.sleep(0.5)

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)

    available_endpoints = [r for r in results if r["available"]]
    unavailable_endpoints = [r for r in results if not r["available"]]

    print(f"\n✅ ДОСТУПНЫЕ ЭНДПОИНТЫ ({len(available_endpoints)}):")
    for ep in available_endpoints:
        count_str = f" ({ep['count']} записей)" if ep.get("count") is not None else ""
        print(f"  - {ep['name']}: {ep['endpoint']}{count_str}")

    print(f"\n❌ НЕДОСТУПНЫЕ ЭНДПОИНТЫ ({len(unavailable_endpoints)}):")
    for ep in unavailable_endpoints:
        print(f"  - {ep['name']}: {ep['endpoint']} (Status: {ep['status_code']})")

    # Сохранить результаты
    out_path = "missing_endpoints_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Полные результаты сохранены в {out_path}")

    # ============================================================================
    # RECOMMENDATIONS
    # ============================================================================

    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)

    names_available = {ep["name"] for ep in available_endpoints}

    if {"Virtual Weather Stations", "Weather History"} & names_available:
        print("\n🌦️  КРИТИЧНО: найдены метеоданные!")
        print("   → Выгрузи weather data для всех полей и лет;")
        print("   → добавь сезонные агрегаты (температура, осадки) в ML-дataset.")

    if {"Soil Samples", "Soil Maps"} & names_available:
        print("\n🌱 Найдены почвенные данные!")
        print("   → Выгрузи pH, NPK, органику и т.п.;")
        print("   → добавь как field-level фичи (статичные по полю).")

    if "History Items" in names_available:
        print("\n📜 Найдена полная история полей!")
        print("   → Можно получить историю культур/урожайности за все годы;")
        print("   → проверь, нет ли более ранних лет, чем в productivity_estimates.")

    if "Fertilizers" in names_available:
        print("\n🧪 Найден справочник удобрений!")
        print("   → Выгрузи точный NPK-состав (element_N, element_P2O5, element_K2O и т.д.);")
        print("   → замени упрощённый 30/10/10 на реальный NPK по applicable_id.")

    print("\n" + "=" * 80)

