"""
Тестирование Cropwise Operations API для получения информации о семенах (applicables).
Цель: найти правильный эндпоинт и структуру данных для маппинга seed_id → crop_id.

Запуск (PowerShell):
  $env:CROPWISE_API_KEY="..."; python test_applicables_api.py

Важно:
  Cropwise Operations API v3 использует заголовок X-User-Api-Token (не Bearer).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

# ⚠️ ВАЖНО: держи токен в env var CROPWISE_API_KEY (не хардкодь в файл)
API_TOKEN = os.getenv("CROPWISE_API_KEY", "your_token_here")

BASE_URL = "https://operations.cropwise.com/api/v3"

# Возможные эндпоинты (будем тестировать все)
# Некоторые могут не существовать в твоем контракте/версии — это нормально.
ENDPOINTS_TO_TEST = [
    "/applicables/{id}",
    "/seeds/{id}",
    "/application_mix_items/{id}",
    "/crops/{id}",
    "/agro_operations/{id}",
    "/history_items/{id}",
]

# Тестовые seed_id (из диагностики, топ-5)
TEST_SEED_IDS = [39, 37, 28, 49, 50]

OUT_JSON = "api_test_results.json"

TIMEOUT_SEC = 15
SLEEP_SEC = 0.4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


@dataclass
class TestResult:
    endpoint: str
    item_id: int
    url: str
    status_code: int | None
    success: bool
    data: Any | None
    error: str | None


def _headers(token: str) -> dict[str, str]:
    # Cropwise Operations v3 auth
    return {
        "X-User-Api-Token": token,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }


def test_endpoint(endpoint_template: str, item_id: int, token: str) -> TestResult:
    url = f"{BASE_URL}{endpoint_template.replace('{id}', str(item_id))}"
    try:
        resp = requests.get(url, headers=_headers(token), timeout=TIMEOUT_SEC)
        ok = resp.status_code == 200
        data = None
        err = None
        if ok:
            try:
                data = resp.json()
            except Exception:
                data = resp.text
        else:
            err = resp.text
        return TestResult(
            endpoint=endpoint_template,
            item_id=item_id,
            url=url,
            status_code=resp.status_code,
            success=ok,
            data=data,
            error=err,
        )
    except Exception as e:
        return TestResult(
            endpoint=endpoint_template,
            item_id=item_id,
            url=url,
            status_code=None,
            success=False,
            data=None,
            error=str(e),
        )


def _find_crop_like_id(data: Any) -> Any | None:
    """
    Грубый поиск crop_id / related_crop_id / base_crop_id в JSON-ответе.
    """
    if not isinstance(data, dict):
        return None
    for key in ("crop_id", "related_crop_id", "base_crop_id"):
        if key in data and data.get(key) not in (None, "", []):
            return data.get(key)
    crop = data.get("crop")
    if isinstance(crop, dict) and crop.get("id") is not None:
        return crop.get("id")
    return None


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ CROPWISE OPERATIONS API ДЛЯ SEED→CROP МАППИНГА")
    print("=" * 80)

    # Проверка токена
    if not API_TOKEN or API_TOKEN == "your_token_here":
        print("\n❌ ОШИБКА: Не указан API токен!")
        print("   Варианты:")
        print('   - PowerShell: $env:CROPWISE_API_KEY="..."; python test_applicables_api.py')
        print("   - или вставь токен в переменную API_TOKEN в этом файле")
        raise SystemExit(1)

    print(f"\n✅ Токен настроен (длина: {len(API_TOKEN)} символов)")
    print(f"   База: {BASE_URL}")
    print(f"   Тестируем {len(ENDPOINTS_TO_TEST)} эндпоинтов на {len(TEST_SEED_IDS)} seed_id")

    results: list[dict[str, Any]] = []

    for endpoint_template in ENDPOINTS_TO_TEST:
        print("\n" + "=" * 80)
        print(f"ТЕСТИРОВАНИЕ: {endpoint_template}")
        print("=" * 80)

        for seed_id in TEST_SEED_IDS:
            print(f"\n  Тестируем id={seed_id}...")
            r = test_endpoint(endpoint_template, seed_id, API_TOKEN)

            row = {
                "endpoint": r.endpoint,
                "seed_id": r.item_id,
                "url": r.url,
                "status_code": r.status_code,
                "success": r.success,
                "data": r.data,
                "error": r.error,
            }
            results.append(row)

            if r.success:
                print(f"    ✅ Успех! Status: {r.status_code}")
                try:
                    data_str = json.dumps(r.data, indent=2, ensure_ascii=False)
                except Exception:
                    data_str = str(r.data)
                print("    Данные (первые 500 символов):")
                print(f"    {data_str[:500]}...")

                crop_like = _find_crop_like_id(r.data)
                if crop_like is not None:
                    print(f"    🎯 НАЙДЕН crop-like id: {crop_like}")
            else:
                print(f"    ❌ Ошибка! Status: {r.status_code}")
                msg = (r.error or "Unknown").replace("\n", " ")
                print(f"    Сообщение: {msg[:200]}")

            time.sleep(SLEEP_SEC)

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)

    # Статистика по эндпоинтам
    working_endpoints: set[str] = set()
    for endpoint_template in ENDPOINTS_TO_TEST:
        endpoint_results = [r for r in results if r["endpoint"] == endpoint_template]
        success_count = sum(1 for r in endpoint_results if r["success"])

        print(f"\n{endpoint_template}:")
        print(f"  Успешных запросов: {success_count}/{len(endpoint_results)}")

        if success_count > 0:
            working_endpoints.add(endpoint_template)
            print("  ✅ РАБОТАЕТ! Используй этот эндпоинт/объект")
            sample = next(r for r in endpoint_results if r["success"])
            print(f"\n  Пример ответа для id={sample['seed_id']}:")
            try:
                print(json.dumps(sample["data"], indent=2, ensure_ascii=False)[:800])
            except Exception:
                print(str(sample["data"])[:800])
            print("\n  ...")
        else:
            print("  ❌ Не работает")

    # Сохранить результаты
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Полные результаты сохранены в {OUT_JSON}")

    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)

    if working_endpoints:
        print(f"\n✅ Рабочие эндпоинты найдены: {sorted(working_endpoints)}")
        print("\nСледующий шаг:")
        print("  1) Если нашёлся endpoint для Seed/Applicables — выгружаем все уникальные applicable_id из mix_items")
        print("  2) Строим маппинг seed_applicable_id → crop_id (или → crop_name/standard_name)")
        print("  3) Джойним в build_ml_dataset.py")
    else:
        print("\n⚠️  Рабочие эндпоинты НЕ найдены")
        print("\nЭто может значить:")
        print("  - эндпоинты другие (в доке есть отношения Crop↔Seeds, но ресурс Seeds может жить иначе)")
        print("  - у токена нет доступа (403/401)")
        print("  - seed_id из mix_items — это ID другого справочника (не crops/seeds)")
        print("\nТогда следующий шаг: искать правильный справочник по документации и пробовать list endpoints:")
        print("  - /seeds (collection), /applicables (collection) и т.п. (если поддерживается контрактом)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

