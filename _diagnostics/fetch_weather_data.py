"""
Выгрузить погодные данные через Virtual Weather Stations / Virtual Weather History Items.
Цель: получить температуру/осадки по station×date и потом замэппить к полям.
"""

import os
import time
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

API_TOKEN = os.getenv("CROPWISE_API_KEY", "your_token_here")  # <-- УСТАНОВИ CROPWISE_API_KEY
BASE_URL = "https://operations.cropwise.com/api/v3"

FIELDS_CSV = "data_raw/fields.csv"

WEATHER_STATIONS_CSV = "data_raw/weather_stations.csv"
WEATHER_HISTORY_CSV = "data_raw/weather_history.csv"
FIELD_STATION_MAP_CSV = "data_raw/field_to_station_mapping.csv"

# Годы для выгрузки погоды
START_YEAR = 2010
END_YEAR = 2025


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "X-User-Api-Token": API_TOKEN,
            "Accept": "application/json",
        }
    )
    return s


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Расстояние между двумя точками в км."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def fetch_virtual_weather_stations(session: requests.Session) -> pd.DataFrame:
    print("\n1) ВЫГРУЗКА МЕТЕОСТАНЦИЙ (`/virtual_weather_stations`)")
    url = f"{BASE_URL}/virtual_weather_stations"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame(data)
    print(f"  ✅ Получено станций: {len(df)}")
    print(f"  Колонки: {list(df.columns)}")
    df.to_csv(WEATHER_STATIONS_CSV, index=False)
    print(f"  💾 Сохранено: {WEATHER_STATIONS_CSV}")
    return df


def map_fields_to_stations(fields_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    print("\n2) МАППИНГ POLY → БЛИЖАЙШАЯ МЕТЕОСТАНЦИЯ")
    if fields_df.empty or stations_df.empty:
        print("  ⚠️ fields или stations пустые — маппинг не делаем")
        return pd.DataFrame()

    required_field_cols = {"id", "lat", "long"}
    if not required_field_cols.issubset(fields_df.columns):
        print(f"  ⚠️ В fields не хватает колонок {required_field_cols} — маппинг не делаем")
        return pd.DataFrame()

    required_station_cols = {"id", "latitude", "longitude"}
    if not required_station_cols.issubset(stations_df.columns):
        print(f"  ⚠️ В stations не хватает колонок {required_station_cols} — маппинг не делаем")
        return pd.DataFrame()

    rows = []
    for _, field in fields_df.iterrows():
        field_id = field["id"]
        field_lat = field["lat"]
        field_lon = field["long"]
        if pd.isna(field_lat) or pd.isna(field_lon):
            continue

        min_dist = float("inf")
        closest_station_id = None
        for _, st in stations_df.iterrows():
            sid = st["id"]
            slat = st["latitude"]
            slon = st["longitude"]
            if pd.isna(slat) or pd.isna(slon):
                continue
            d = haversine_distance(field_lat, field_lon, slat, slon)
            if d < min_dist:
                min_dist = d
                closest_station_id = sid

        if closest_station_id is not None:
            rows.append(
                {
                    "field_id": field_id,
                    "station_id": closest_station_id,
                    "distance_km": round(min_dist, 3),
                }
            )

    mapping_df = pd.DataFrame(rows)
    if mapping_df.empty:
        print("  ⚠️ Не удалось построить маппинг field→station")
        return mapping_df

    print("  ✅ Маппинг создан")
    print(f"  Средняя дистанция: {mapping_df['distance_km'].mean():.2f} км")
    print(f"  Макс. дистанция:   {mapping_df['distance_km'].max():.2f} км")
    mapping_df.to_csv(FIELD_STATION_MAP_CSV, index=False)
    print(f"  💾 Сохранено: {FIELD_STATION_MAP_CSV}")
    return mapping_df


def fetch_weather_history_for_station(
    session: requests.Session, station_id: int, start_year: int, end_year: int
) -> list[dict]:
    """
    Пытаемся вытащить историю погоды по станции через /virtual_weather_history_items.
    Формат по докам: GET /virtual_weather_history_items?virtual_weather_station_id=&from_date=&to_date=
    """
    records: list[dict] = []
    print(f"\n   Станция {station_id}: выгрузка {start_year}-{end_year}")

    for year in range(start_year, end_year + 1):
        from_date = f"{year}-01-01"
        to_date = f"{year}-12-31"
        params = {
            "virtual_weather_station_id": station_id,
            "from_date": from_date,
            "to_date": to_date,
        }
        url = f"{BASE_URL}/virtual_weather_history_items"
        try:
            resp = session.get(url, params=params, timeout=40)
        except Exception as e:
            print(f"     {year}: ❌ запрос упал: {e}")
            continue

        if resp.status_code != 200:
            print(f"     {year}: ❌ {resp.status_code} ({resp.text[:120]})")
            # если эндпоинта вообще нет — выходим, чтобы не долбить зря
            if resp.status_code == 404:
                break
            continue

        data = resp.json()
        if isinstance(data, dict):
            items = data.get("data") or data.get("items") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for it in items:
            it["station_id"] = station_id
        records.extend(items)
        print(f"     {year}: ✅ {len(items)} записей")
        time.sleep(0.3)

    return records


def main() -> None:
    print("=" * 80)
    print("ВЫГРУЗКА ПОГОДНЫХ ДАННЫХ")
    print("=" * 80)

    if API_TOKEN == "your_token_here":
        print("\n❌ ОШИБКА: Не указан API токен (CROPWISE_API_KEY)!")
        return

    session = get_session()

    # 1) Станции
    stations_df = fetch_virtual_weather_stations(session)

    # 2) Маппинг полей к станциям
    if not os.path.exists(FIELDS_CSV):
        print(f"\n⚠️ Файл с полями не найден: {FIELDS_CSV}")
        return
    fields_df = pd.read_csv(FIELDS_CSV)
    mapping_df = map_fields_to_stations(fields_df, stations_df)

    # 3) История погоды по уникальным станциям
    print(f"\n3) ВЫГРУЗКА ПОГОДНОЙ ИСТОРИИ ({START_YEAR}-{END_YEAR})")
    if mapping_df.empty:
        print("  ⚠️ Нет маппинга field→station, выгружаем погоду для всех станций без привязки.")
        station_ids = stations_df["id"].unique().tolist()
    else:
        station_ids = sorted(mapping_df["station_id"].unique().tolist())

    print(f"  Уникальных станций для выгрузки: {len(station_ids)}")
    all_weather: list[dict] = []
    for sid in station_ids:
        recs = fetch_weather_history_for_station(session, sid, START_YEAR, END_YEAR)
        all_weather.extend(recs)

    if not all_weather:
        print("\n❌ Не удалось получить погодные данные (проверь, доступен ли `/virtual_weather_history_items`)")
    else:
        weather_df = pd.DataFrame(all_weather)
        # если есть поле date/datetime — привести к дате
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        elif "datetime" in weather_df.columns:
            weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], errors="coerce")
        weather_df.to_csv(WEATHER_HISTORY_CSV, index=False)
        print(f"\n✅ Сохранено {len(weather_df)} записей погоды")
        print(f"💾 Файл: {WEATHER_HISTORY_CSV}")
        print(f"Колонки: {list(weather_df.columns)}")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()

