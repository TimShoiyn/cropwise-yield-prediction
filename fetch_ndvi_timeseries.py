"""
Скрипт для выгрузки NDVI временных рядов из Cropwise Operations API.

Использует эндпоинт /api/v3a/historical_values?type=ndvi (Operations API),
который доступен с тем же токеном, что и остальные запросы.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Sequence

import pandas as pd
import requests
from requests.exceptions import HTTPError
from urllib.parse import urljoin


# ============================================================================
# CONFIGURATION
# ============================================================================

# Токен Operations API (тот же, что используется в fetch_cropwise_data.py)
API_KEY = os.getenv("CROPWISE_API_KEY", "")

# Base URL Operations API
BASE_URL = "https://operations.cropwise.com/api/v3/"
API_KEY_HEADER_NAME = "X-User-Api-Token"
API_KEY_HEADER_PREFIX = ""

# Input files
DATA_DIR = "data_raw"
FIELDS_CSV = os.path.join(DATA_DIR, "fields.csv")
PRODUCTIVITY_CSV = os.path.join(DATA_DIR, "productivity_estimates.csv")
OPERATIONS_CSV = os.path.join(DATA_DIR, "operations.csv")

# Output file
NDVI_OUTPUT = os.path.join(DATA_DIR, "ndvi_timeseries.csv")

# Rate limiting
REQUEST_DELAY = 0.3
REQUEST_TIMEOUT = 30


# ============================================================================
# HTTP CLIENT
# ============================================================================

class CropwiseClient:
    """Клиент для Operations API (тот же, что в fetch_cropwise_data.py)"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        header_name: str = API_KEY_HEADER_NAME,
        header_prefix: str = API_KEY_HEADER_PREFIX,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
            raise ValueError("API key is empty. Задай CROPWISE_API_KEY в переменных окружения.")
        
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            header_name: f"{header_prefix}{api_key}",
        })
    
    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))
    
    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET запрос с обработкой ошибок"""
        url = self._url(path)
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        
        if not resp.ok:
            resp.raise_for_status()
        
        return resp.json()


def _as_list(obj: Any) -> List[Any]:
    """Привести ответ API к списку"""
    if obj is None:
        return []
    
    if isinstance(obj, list):
        return obj
    
    if isinstance(obj, dict):
        for key in ("data", "items", "results"):
            if key in obj and isinstance(obj[key], Sequence):
                return list(obj[key])
        return [obj]
    
    return []


# ============================================================================
# NDVI FETCHING
# ============================================================================

def get_field_ndvi_history(
    client: CropwiseClient,
    field_id: int,
    *,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    source_sign: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Исторические значения NDVI по полю из Operations API.
    
    Эндпоинт: GET /api/v3a/historical_values
    Параметры:
      - field_id (обязательно)
      - type=ndvi (обязательно)
      - from_time, to_time (опционально, формат ISO или YYYY-MM-DD)
      - source_sign (опционально)
    """
    params: Dict[str, Any] = {
        "field_id": field_id,
        "type": "ndvi"
    }
    
    if from_time:
        params["from_time"] = from_time
    if to_time:
        params["to_time"] = to_time
    if source_sign:
        params["source_sign"] = source_sign
    
    try:
        # Используем historical_values эндпоинт
        # В fetch_cropwise_data.py используется /historical_values
        # Попробуем сначала v3a путь (полный URL), если не сработает - обычный
        try:
            # Для v3a используем полный URL
            url = "https://operations.cropwise.com/api/v3a/historical_values"
            resp = client.session.get(url, params=params, timeout=client.timeout)
            if not resp.ok:
                resp.raise_for_status()
            raw = resp.json()
        except (HTTPError, Exception):
            # Fallback на обычный путь через клиент (как в fetch_cropwise_data.py)
            raw = client.get("/historical_values", params=params)
        return _as_list(raw)
    except HTTPError as e:
        if e.response is not None:
            status_code = e.response.status_code
            if status_code == 404:
                # Эндпоинт может не существовать для некоторых аккаунтов
                return []
            elif status_code in [401, 403]:
                # Проблема с доступом
                raise
        raise


def get_season_dates(field_id: int, year: int, operations_df: pd.DataFrame) -> tuple[str, str]:
    """
    Извлекает даты сезона из операций для поля и года.
    Если операций нет - использует стандартные даты вегетационного сезона.
    """
    field_ops = operations_df[
        (operations_df['field_id'] == field_id) &
        (operations_df['season'] == str(year))
    ].copy()
    
    if len(field_ops) == 0:
        start_date = f"{year}-03-01"
        end_date = f"{year}-10-31"
        return start_date, end_date
    
    date_cols = ['planned_start_date', 'actual_start_datetime', 'completed_date']
    all_dates = []
    
    for col in date_cols:
        if col in field_ops.columns:
            dates = pd.to_datetime(field_ops[col], errors='coerce').dropna()
            all_dates.extend(dates.tolist())
    
    if len(all_dates) == 0:
        start_date = f"{year}-03-01"
        end_date = f"{year}-10-31"
        return start_date, end_date
    
    start_date = min(all_dates).strftime('%Y-%m-%d')
    end_date = max(all_dates).strftime('%Y-%m-%d')
    
    return start_date, end_date


# ============================================================================
# MAIN EXTRACTION LOGIC
# ============================================================================

def main():
    print("=" * 80)
    print("NDVI TIME SERIES EXTRACTION FROM CROPWISE OPERATIONS API")
    print("=" * 80)
    print()
    
    # Проверка токена
    if not API_KEY:
        print("❌ ОШИБКА: Токен не задан!")
        print()
        print("Установи переменную окружения:")
        print("  PowerShell: $env:CROPWISE_API_KEY = 'твой_токен'")
        print()
        return
    
    # Инициализация клиента
    try:
        client = CropwiseClient(API_KEY)
    except ValueError as e:
        print(f"❌ ОШИБКА: {e}")
        return
    
    # Загрузка данных
    print("📂 Загрузка локальных CSV файлов...")
    try:
        fields_df = pd.read_csv(FIELDS_CSV)
        productivity_df = pd.read_csv(PRODUCTIVITY_CSV)
        operations_df = pd.read_csv(OPERATIONS_CSV)
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print(f"   Убедись, что файлы находятся в {DATA_DIR}/")
        return
    
    print(f"  ✅ Загружено {len(fields_df)} полей")
    print(f"  ✅ Загружено {len(productivity_df)} оценок урожайности")
    print(f"  ✅ Загружено {len(operations_df)} операций")
    print()
    
    # Подготовка выходных данных
    ndvi_records = []
    
    # Итерация по каждой оценке урожайности (поле × год)
    total = len(productivity_df)
    success_count = 0
    fail_count = 0
    
    # Для теста ограничиваем первыми 3 записями
    TEST_MODE = False  # Поставь True для теста
    max_records = 3 if TEST_MODE else total
    
    print(f"🌾 Обработка {min(max_records, total)} комбинаций поле×год...")
    if TEST_MODE:
        print("  ⚠️  ТЕСТОВЫЙ РЕЖИМ: обрабатываются только первые 3 записи")
    print()
    
    for idx, row in productivity_df.iterrows():
        if idx >= max_records:
            break
        field_id = int(row['field_id'])
        year = int(row['year'])
        
        print(f"[{idx+1}/{total}] Поле {field_id}, Год {year}")
        
        # Получаем даты сезона
        start_date, end_date = get_season_dates(field_id, year, operations_df)
        print(f"  📅 Сезон: {start_date} → {end_date}")
        
        # Запрашиваем NDVI временной ряд
        print(f"  🛰️  Запрос NDVI из Operations API...")
        try:
            ndvi_items = get_field_ndvi_history(
                client,
                field_id,
                from_time=start_date,
                to_time=end_date,
            )
        except HTTPError as e:
            if e.response is not None:
                status_code = e.response.status_code
                if status_code == 404:
                    print(f"  ⚠️  Эндпоинт не найден (404), пропускаю")
                    fail_count += 1
                    continue
                elif status_code in [401, 403]:
                    print(f"  ❌ Ошибка доступа ({status_code})")
                    print(f"     Проверь токен и права доступа")
                    fail_count += 1
                    continue
            print(f"  ❌ HTTP ошибка: {e}")
            fail_count += 1
            continue
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            fail_count += 1
            continue
        
        if len(ndvi_items) == 0:
            print(f"  ⚠️  Нет данных NDVI для этого периода")
            fail_count += 1
            time.sleep(REQUEST_DELAY)
            continue
        
        print(f"  ✅ Получено {len(ndvi_items)} записей от API")
        
        # Дебаг: показываем структуру первых записей
        if len(ndvi_items) > 0:
            print(f"  🔍 Дебаг: анализирую структуру ответа...")
            for i, item in enumerate(ndvi_items[:3]):  # Первые 3 записи
                print(f"  🔍 Запись {i+1}: ключи = {list(item.keys())[:15]}")
                if 'product_type' in item:
                    print(f"      product_type = {item.get('product_type')}")
                for key in ['ndvi', 'value', 'time_series', 'values']:
                    if key in item:
                        val = item.get(key)
                        print(f"      {key} = {type(val).__name__}, длина = {len(val) if isinstance(val, (list, dict)) else 'N/A'}")
                        if isinstance(val, list) and len(val) > 0:
                            print(f"         первый элемент = {val[0]}")
            print()
        
        # Фильтруем только NDVI данные и разворачиваем временные ряды
        ndvi_count = 0
        for item in ndvi_items:
            # Пропускаем не-NDVI данные (если есть product_type)
            product_type = item.get('product_type', '').lower()
            if product_type and product_type != 'ndvi':
                continue
            
            # Если есть прямое поле date и ndvi (без product_type)
            if 'date' in item and 'ndvi' in item and not product_type:
                date_val = item.get('date')
                ndvi_val = item.get('ndvi')
                try:
                    ndvi_float = float(ndvi_val)
                    record = {
                        'field_id': field_id,
                        'year': year,
                        'date': date_val,
                        'ndvi_mean': ndvi_float,
                        'ndvi_min': None,
                        'ndvi_max': None,
                        'ndvi_std': None,
                        'source': item.get('source') or item.get('source_sign', 'unknown'),
                        'cloud_coverage': item.get('cloud_coverage'),
                        'data_coverage': item.get('data_coverage'),
                    }
                    ndvi_records.append(record)
                    ndvi_count += 1
                    continue
                except (TypeError, ValueError):
                    pass
            
            # Пробуем разные поля для NDVI данных
            ndvi_data = (
                item.get('ndvi') or 
                item.get('value') or 
                item.get('index_value') or
                item.get('time_series') or
                (item.get('values') if isinstance(item.get('values'), list) else None)
            )
            
            if ndvi_data is None:
                continue
            
            # Если ndvi - это массив временного ряда [[date, value], ...]
            if isinstance(ndvi_data, list) and len(ndvi_data) > 0:
                # Проверяем формат: первый элемент - это [date, value] или просто value?
                first_elem = ndvi_data[0]
                if isinstance(first_elem, list) and len(first_elem) >= 2:
                    # Формат: [[date, value], [date, value], ...]
                    for point in ndvi_data:
                        if isinstance(point, list) and len(point) >= 2:
                            date_str = point[0]
                            value = point[1]
                            
                            # Если value - это список (например, для soil_moisture), пропускаем
                            if isinstance(value, list):
                                continue
                            
                            try:
                                ndvi_val = float(value)
                            except (TypeError, ValueError):
                                continue
                            
                            record = {
                                'field_id': field_id,
                                'year': year,
                                'date': date_str,
                                'ndvi_mean': ndvi_val,
                                'ndvi_min': None,
                                'ndvi_max': None,
                                'ndvi_std': None,
                                'source': item.get('source') or item.get('source_sign', 'unknown'),
                                'cloud_coverage': None,
                                'data_coverage': None,
                            }
                            ndvi_records.append(record)
                            ndvi_count += 1
                else:
                    # Формат: [value1, value2, ...] - нет дат, пропускаем
                    continue
            elif isinstance(ndvi_data, (int, float)):
                # Если ndvi - это одно число, используем last_value_date как дату
                record = {
                    'field_id': field_id,
                    'year': year,
                    'date': item.get('last_value_date'),
                    'ndvi_mean': float(ndvi_data),
                    'ndvi_min': None,
                    'ndvi_max': None,
                    'ndvi_std': None,
                    'source': item.get('source') or item.get('source_sign', 'unknown'),
                    'cloud_coverage': None,
                    'data_coverage': None,
                }
                ndvi_records.append(record)
                ndvi_count += 1
        
        if ndvi_count > 0:
            print(f"  ✅ Извлечено {ndvi_count} точек NDVI из временных рядов")
        else:
            print(f"  ⚠️  Не найдено NDVI данных в ответе API")
        
        success_count += 1
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
        print()
    
    # ========================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================================
    
    print("=" * 80)
    print("📊 ИЗВЛЕЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"✅ Успешно: {success_count} комбинаций поле×год")
    print(f"❌ Провалено: {fail_count} комбинаций")
    print(f"📈 Всего наблюдений NDVI: {len(ndvi_records)}")
    print()
    
    if len(ndvi_records) == 0:
        print("⚠️  ВНИМАНИЕ: Данные NDVI не получены!")
        print()
        print("ВОЗМОЖНЫЕ ПРИЧИНЫ:")
        print("1. Эндпоинт /api/v3a/historical_values недоступен для твоего аккаунта")
        print("2. Нет спутникового покрытия для этих полей/дат")
        print("3. Неправильный формат дат или параметров")
        print()
        print("ПОПРОБУЙ:")
        print("→ Проверь, что эндпоинт доступен в документации Operations API")
        print("→ Попробуй запустить fetch_cropwise_data.py с FETCH_NDVI_HISTORY=True")
        print()
        return
    
    # Сохранение в CSV
    ndvi_df = pd.DataFrame(ndvi_records)
    ndvi_df.to_csv(NDVI_OUTPUT, index=False)
    
    print(f"💾 Сохранено в: {NDVI_OUTPUT}")
    print()
    
    # Показываем пример
    print("📋 Пример извлеченных данных:")
    print(ndvi_df.head(10).to_string())
    print()
    
    # Статистика
    if len(ndvi_df) > 0 and 'ndvi_mean' in ndvi_df.columns:
        print("📊 Статистика NDVI:")
        # Фильтруем только числовые значения (не списки)
        numeric_ndvi = pd.to_numeric(ndvi_df['ndvi_mean'], errors='coerce')
        if numeric_ndvi.notna().sum() > 0:
            stats = ndvi_df.groupby('field_id').apply(
                lambda x: pd.Series({
                    'count': len(x),
                    'mean': pd.to_numeric(x['ndvi_mean'], errors='coerce').mean(),
                    'min': pd.to_numeric(x['ndvi_mean'], errors='coerce').min(),
                    'max': pd.to_numeric(x['ndvi_mean'], errors='coerce').max()
                })
            )
            print(stats.head(20))
        else:
            print("  ⚠️  Нет числовых значений NDVI для статистики")
        print()
    
    print("=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
