"""
Выгрузка погодной истории (WeatherHistoryItems) из Cropwise Operations API.

Что делает:
- Для всех field_group_id из data_raw/fields.csv:
    - дергает /api/v3/weather_history_items?field_group_id=...&from_time=...&to_time=...
      по годам в заданном диапазоне;
    - собирает ежедневные ряды температур/осадков/снега;
    - сохраняет «сырые» ряды в data_raw/weather_history_items.csv;
    - дополнительно агрегирует по (field_group_id, year) → сезонные фичи
      и маппит их обратно на field_id → data_raw/weather_season_aggregates.csv.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from fetch_cropwise_data import CropwiseClient, API_KEY, BASE_URL


DATA_DIR = "data_raw"
os.makedirs(DATA_DIR, exist_ok=True)

FIELDS_CSV = os.path.join(DATA_DIR, "fields.csv")
WEATHER_HISTORY_RAW_CSV = os.path.join(DATA_DIR, "weather_history_items.csv")
WEATHER_SEASON_AGG_CSV = os.path.join(DATA_DIR, "weather_season_aggregates.csv")


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


def detect_year_range_from_targets() -> tuple[int, int]:
    """
    Пытаемся ограничить годы по targets_extended или productivity_estimates.
    Если не получилось — возвращаем (2010, текущий год).
    """
    now_year = datetime.utcnow().year
    candidates: List[int] = []
    targets_path = os.path.join("data_processed", "targets_extended.csv")
    if os.path.exists(targets_path):
        try:
            tdf = pd.read_csv(targets_path)
            if "year" in tdf.columns:
                years = pd.to_numeric(tdf["year"], errors="coerce").dropna().astype(int)
                if not years.empty:
                    return int(years.min()), int(years.max())
        except Exception:
            pass

    prod_path = os.path.join(DATA_DIR, "productivity_estimates.csv")
    if os.path.exists(prod_path):
        try:
            pdf = pd.read_csv(prod_path)
            if "year" in pdf.columns:
                years = pd.to_numeric(pdf["year"], errors="coerce").dropna().astype(int)
                if not years.empty:
                    return int(years.min()), int(years.max())
        except Exception:
            pass

    # фоллбек
    return 2010, now_year


def fetch_weather_history_items() -> pd.DataFrame:
    """
    GET /api/v3/weather_history_items для всех field_group_id × years.
    """
    print("=" * 80)
    print("ВЫГРУЗКА ПОГОДНЫХ ДАННЫХ (WeatherHistoryItems)")
    print("=" * 80)

    _ensure_api_key()

    if not os.path.exists(FIELDS_CSV):
        raise FileNotFoundError(f"{FIELDS_CSV} не найден (нужен для field_group_id)")

    fields_df = pd.read_csv(FIELDS_CSV)
    if "field_group_id" not in fields_df.columns:
        raise ValueError("В fields.csv нет колонки field_group_id")

    field_groups = (
        fields_df["field_group_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    print(f"Найдено field_group_id: {field_groups}")

    year_min, year_max = detect_year_range_from_targets()
    years = list(range(year_min, year_max + 1))
    print(f"Диапазон лет для выгрузки погоды: {year_min}–{year_max}")

    client = CropwiseClient(API_KEY, base_url=BASE_URL)

    all_records: List[Dict[str, Any]] = []

    for fg in field_groups:
        for year in years:
            from_time = f"{year}-01-01"
            to_time = f"{year}-12-31"
            params = {
                "field_group_id": fg,
                "from_time": from_time,
                "to_time": to_time,
            }
            print(f"  -> field_group_id={fg}, year={year} ...")
            try:
                resp = client.get("/weather_history_items", params=params)
            except Exception as e:
                print(f"     ⚠️ Ошибка {e}, пропускаю этот год/field_group")
                continue

            items = _as_list(resp)
            if not items:
                print("     (пусто)")
                continue

            for it in items:
                rec = dict(it)
                rec["field_group_id"] = fg
                # контрольный дублирующий год, если date есть
                if "date" in rec and rec["date"]:
                    try:
                        d = pd.to_datetime(rec["date"])
                        rec["year"] = int(d.year)
                    except Exception:
                        rec["year"] = year
                else:
                    rec["year"] = year
                all_records.append(rec)

    if not all_records:
        print("⚠️ Не получили ни одной записи погоды (all_records пустой)")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"\n✅ Получено записей погоды: {len(df)}")
    print("Колонки:", list(df.columns))

    df.to_csv(WEATHER_HISTORY_RAW_CSV, index=False)
    print(f"💾 Сырые погодные ряды сохранены в: {WEATHER_HISTORY_RAW_CSV}")
    return df


def build_season_aggregates(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит сезонные агрегаты по (field_group_id, year) и маппит их к field_id.
    """
    print("\n" + "=" * 80)
    print("АГРЕГАЦИЯ ПОГОДЫ ПО СЕЗОНУ (field_group_id × year)")
    print("=" * 80)

    if weather_df.empty:
        print("⚠️ weather_df пустой, агрегаты не строим")
        return pd.DataFrame()

    # Гарантируем числовые типы для базовых метрик
    num_cols = [
        "temperature_min",
        "temperature_avg",
        "temperature_max",
        "precipitation",
        "snow",
    ]
    for col in num_cols:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

    group_cols = ["field_group_id", "year"]
    agg_dict: Dict[str, Any] = {}
    if "temperature_min" in weather_df.columns:
        agg_dict["temp_min_mean"] = ("temperature_min", "mean")
        agg_dict["temp_min_min"] = ("temperature_min", "min")
        agg_dict["temp_min_max"] = ("temperature_min", "max")
    if "temperature_avg" in weather_df.columns:
        agg_dict["temp_avg_mean"] = ("temperature_avg", "mean")
        agg_dict["temp_avg_min"] = ("temperature_avg", "min")
        agg_dict["temp_avg_max"] = ("temperature_avg", "max")
    if "temperature_max" in weather_df.columns:
        agg_dict["temp_max_mean"] = ("temperature_max", "mean")
        agg_dict["temp_max_min"] = ("temperature_max", "min")
        agg_dict["temp_max_max"] = ("temperature_max", "max")
    if "precipitation" in weather_df.columns:
        agg_dict["precip_sum"] = ("precipitation", "sum")
        agg_dict["precip_mean"] = ("precipitation", "mean")
    if "snow" in weather_df.columns:
        agg_dict["snow_mean"] = ("snow", "mean")
        agg_dict["snow_max"] = ("snow", "max")

    grouped = (
        weather_df.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    print(f"✅ Сезонных записей по погоде: {len(grouped)}")

    # Маппим field_group_id обратно к field_id (может быть несколько полей в группе)
    fields_df = pd.read_csv(FIELDS_CSV)
    fg_to_fields = (
        fields_df[["field_group_id", "id"]]
        .rename(columns={"id": "field_id"})
        .dropna(subset=["field_group_id"])
    )
    fg_to_fields["field_group_id"] = fg_to_fields["field_group_id"].astype(int)

    merged = grouped.merge(fg_to_fields, on="field_group_id", how="left")

    merged.to_csv(WEATHER_SEASON_AGG_CSV, index=False)
    print(f"💾 Сезонные агрегаты погоды сохранены в: {WEATHER_SEASON_AGG_CSV}")
    return merged


def main() -> None:
    weather_df = fetch_weather_history_items()
    if not weather_df.empty:
        build_season_aggregates(weather_df)
    print("\n" + "=" * 80)
    print("✅ ВЫГРУЗКА ПОГОДЫ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

