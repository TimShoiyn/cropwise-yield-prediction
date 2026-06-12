"""
Полная инвентаризация всех данных из Cropwise API.
Цель: понять, сколько данных доступно и что ещё можно вытащить.
"""

import os
from datetime import datetime

import pandas as pd


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️ Файл не найден: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Не удалось прочитать {path}: {e}")
        return pd.DataFrame()


print("=" * 80)
print("ПОЛНАЯ ИНВЕНТАРИЗАЦИЯ ДАННЫХ CROPWISE")
print("=" * 80)

# ============================================================================
# 1. FIELDS (поля)
# ============================================================================
fields_df = safe_read_csv("data_raw/fields.csv")

print("\n" + "=" * 80)
print("1. FIELDS (ПОЛЯ)")
print("=" * 80)
if fields_df.empty:
    print("❌ data_raw/fields.csv пустой или не найден")
else:
    print(f"Всего полей: {len(fields_df)}")
    print(f"Уникальных field_id: {fields_df['id'].nunique()}")
    print(f"\nID полей: {sorted(fields_df['id'].tolist())}")

    if "created_at" in fields_df.columns:
        created = pd.to_datetime(fields_df["created_at"], errors="coerce")
        print(f"\nДаты создания полей:")
        print(f"  Первое поле создано: {created.min()}")
        print(f"  Последнее поле создано: {created.max()}")
    else:
        print("\nКолонка created_at в fields отсутствует")

# ============================================================================
# 2. OPERATIONS (операции)
# ============================================================================
operations_df = safe_read_csv("data_raw/operations.csv")

print("\n" + "=" * 80)
print("2. OPERATIONS (АГРООПЕРАЦИИ)")
print("=" * 80)
if operations_df.empty:
    print("❌ data_raw/operations.csv пустой или не найден")
else:
    print(f"Всего операций: {len(operations_df)}")
    if "field_id" in operations_df.columns:
        print(f"Уникальных field_id: {operations_df['field_id'].nunique()}")
    if "season" in operations_df.columns:
        print(f"Уникальных season (годов): {operations_df['season'].nunique()}")

        print(f"\nРаспределение по годам (season):")
        season_counts = operations_df["season"].value_counts().sort_index()
        print(season_counts)

        print(f"\nПокрытие field×year в operations:")
        operations_field_years = (
            operations_df.groupby(["field_id", "season"])
            .size()
            .reset_index(name="operations_count")
        )
        print(f"  Уникальных field×year: {len(operations_field_years)}")
        print(
            f"  Минимум операций на field×year: {operations_field_years['operations_count'].min()}"
        )
        print(
            f"  Максимум операций на field×year: {operations_field_years['operations_count'].max()}"
        )
        print(
            f"  Среднее операций на field×year: {operations_field_years['operations_count'].mean():.1f}"
        )
    else:
        operations_field_years = pd.DataFrame()

    if "field_id" in operations_df.columns:
        print(f"\nПоля с операциями:")
        fields_with_ops = sorted(operations_df["field_id"].unique())
        print(f"  Количество: {len(fields_with_ops)}")
        print(f"  ID: {fields_with_ops}")
    else:
        fields_with_ops = []

# ============================================================================
# 3. PRODUCTIVITY ESTIMATES (урожайность - ТАРГЕТ)
# ============================================================================
productivity_df = safe_read_csv("data_raw/productivity_estimates.csv")

print("\n" + "=" * 80)
print("3. PRODUCTIVITY ESTIMATES (УРОЖАЙНОСТЬ - ТАРГЕТ)")
print("=" * 80)
if productivity_df.empty:
    print("❌ data_raw/productivity_estimates.csv пустой или не найден")
else:
    print(f"Всего записей: {len(productivity_df)}")
    if "field_id" in productivity_df.columns:
        print(f"Уникальных field_id: {productivity_df['field_id'].nunique()}")
    if "year" in productivity_df.columns:
        print(f"Уникальных year: {productivity_df['year'].nunique()}")

        print(f"\nРаспределение по годам:")
        year_counts = productivity_df["year"].value_counts().sort_index()
        print(year_counts)

        print(f"\nПокрытие field×year в productivity_estimates:")
        productivity_field_years = (
            productivity_df.groupby(["field_id", "year"])
            .size()
            .reset_index(name="count")
        )
        print(f"  Уникальных field×year: {len(productivity_field_years)}")
    else:
        productivity_field_years = pd.DataFrame()

    if "field_id" in productivity_df.columns:
        print(f"\nПоля с productivity_estimates:")
        fields_with_prod = sorted(productivity_df["field_id"].unique())
        print(f"  Количество: {len(fields_with_prod)}")
        print(f"  ID: {fields_with_prod}")
    else:
        fields_with_prod = []

# ============================================================================
# 4. NDVI TIME SERIES (спутниковые данные)
# ============================================================================
print("\n" + "=" * 80)
print("4. NDVI TIME SERIES (СПУТНИКОВЫЕ ДАННЫЕ)")
print("=" * 80)

ndvi_path = "data_raw/ndvi_timeseries.csv"
if not os.path.exists(ndvi_path):
    print(f"❌ Файл не найден: {ndvi_path}")
    ndvi_df = pd.DataFrame()
else:
    ndvi_file_size = os.path.getsize(ndvi_path) / (1024**2)
    print(f"Размер файла ndvi_timeseries.csv: {ndvi_file_size:.1f} MB")

    print("\nЗагрузка ndvi_timeseries.csv (может занять время)...")
    ndvi_df = pd.read_csv(ndvi_path)

if ndvi_df.empty:
    print("❌ NDVI данные не загружены")
else:
    print(f"Всего NDVI наблюдений: {len(ndvi_df):,}")
    if "field_id" in ndvi_df.columns:
        print(f"Уникальных field_id: {ndvi_df['field_id'].nunique()}")
    if "year" in ndvi_df.columns:
        print(f"Уникальных year: {ndvi_df['year'].nunique()}")

        print(f"\nРаспределение по годам:")
        ndvi_year_counts = ndvi_df["year"].value_counts().sort_index()
        print(ndvi_year_counts)

        print(f"\nПокрытие field×year в NDVI:")
        ndvi_field_years = (
            ndvi_df.groupby(["field_id", "year"])
            .size()
            .reset_index(name="ndvi_obs_count")
        )
        print(f"  Уникальных field×year: {len(ndvi_field_years)}")
        print(
            f"  Минимум наблюдений NDVI на field×year: {ndvi_field_years['ndvi_obs_count'].min()}"
        )
        print(
            f"  Максимум наблюдений NDVI на field×year: {ndvi_field_years['ndvi_obs_count'].max()}"
        )
        print(
            f"  Среднее наблюдений NDVI на field×year: {ndvi_field_years['ndvi_obs_count'].mean():.1f}"
        )
    else:
        ndvi_field_years = pd.DataFrame()

    if "date" in ndvi_df.columns:
        ndvi_df["date"] = pd.to_datetime(ndvi_df["date"], errors="coerce")
        print(f"\nДиапазон дат NDVI:")
        print(f"  Первая дата: {ndvi_df['date'].min()}")
        print(f"  Последняя дата: {ndvi_df['date'].max()}")

    if "field_id" in ndvi_df.columns:
        print(f"\nПоля с NDVI данными:")
        fields_with_ndvi = sorted(ndvi_df["field_id"].unique())
        print(f"  Количество: {len(fields_with_ndvi)}")
        print(f"  ID: {fields_with_ndvi}")
    else:
        fields_with_ndvi = []

# ============================================================================
# 5. YIELD MAPS (карты урожайности)
# ============================================================================
yield_maps_df = safe_read_csv("data_raw/yield_maps.csv")

print("\n" + "=" * 80)
print("5. YIELD MAPS (КАРТЫ УРОЖАЙНОСТИ)")
print("=" * 80)
if yield_maps_df.empty:
    print("❌ data_raw/yield_maps.csv пустой или не найден")
else:
    print(f"Всего yield maps: {len(yield_maps_df)}")
    if "field_id" in yield_maps_df.columns:
        print(f"Уникальных field_id: {yield_maps_df['field_id'].nunique()}")

    if "created_at" in yield_maps_df.columns:
        yield_maps_df["created_at"] = pd.to_datetime(
            yield_maps_df["created_at"], errors="coerce"
        )
        yield_maps_df["year"] = yield_maps_df["created_at"].dt.year
        print(f"\nРаспределение по годам (из created_at):")
        print(yield_maps_df["year"].value_counts().sort_index())

    if "field_id" in yield_maps_df.columns:
        print(f"\nПоля с yield_maps:")
        fields_with_yield_maps = sorted(yield_maps_df["field_id"].unique())
        print(f"  Количество: {len(fields_with_yield_maps)}")
        print(f"  ID: {fields_with_yield_maps}")
    else:
        fields_with_yield_maps = []

# ============================================================================
# 6. СВОДКА: ЧТО ПОКРЫТО, ЧТО НЕТ
# ============================================================================
print("\n" + "=" * 80)
print("6. СВОДНАЯ ТАБЛИЦА ПОКРЫТИЯ")
print("=" * 80)

all_fields = set(fields_df["id"]) if not fields_df.empty else set()
fields_prod = set(fields_with_prod) if "fields_with_prod" in locals() else set()
fields_ops = set(fields_with_ops) if "fields_with_ops" in locals() else set()
fields_ndvi = set(fields_with_ndvi) if "fields_with_ndvi" in locals() else set()
fields_yield = (
    set(fields_with_yield_maps) if "fields_with_yield_maps" in locals() else set()
)

if not all_fields:
    print("❌ Нельзя посчитать покрытие по полям — нет fields")
else:
    print(f"\nПокрытие по полям:")
    print(f"  Всего полей: {len(all_fields)}")
    print(
        f"  С productivity_estimates: {len(fields_prod)} ({len(fields_prod)/len(all_fields)*100:.1f}%)"
    )
    print(
        f"  С operations: {len(fields_ops)} ({len(fields_ops)/len(all_fields)*100:.1f}%)"
    )
    print(
        f"  С NDVI: {len(fields_ndvi)} ({len(fields_ndvi)/len(all_fields)*100:.1f}%)"
    )
    print(
        f"  С yield_maps: {len(fields_yield)} ({len(fields_yield)/len(all_fields)*100:.1f}%)"
    )

    print(f"\nПоля БЕЗ productivity_estimates (не можем использовать для ML):")
    fields_no_prod = all_fields - fields_prod
    if fields_no_prod:
        print(f"  ID: {sorted(fields_no_prod)}")
    else:
        print("  ✅ Все поля имеют productivity_estimates!")

    print(f"\nПоля С productivity_estimates, но БЕЗ operations:")
    fields_prod_no_ops = fields_prod - fields_ops
    if fields_prod_no_ops:
        print(f"  Количество: {len(fields_prod_no_ops)}")
        print(f"  ID: {sorted(fields_prod_no_ops)}")

# ============================================================================
# 7. МАКСИМАЛЬНО ВОЗМОЖНЫЙ ДАТАСЕТ
# ============================================================================
print("\n" + "=" * 80)
print("7. МАКСИМАЛЬНО ВОЗМОЖНЫЙ ДАТАСЕТ")
print("=" * 80)

if not productivity_df.empty and "year" in productivity_df.columns:
    print(f"\nВАРИАНТ 1: Только field×year с productivity_estimates (ТЕКУЩИЙ)")
    print(f"  Датасет: {len(productivity_df)} строк")
    print("  Это ТАРГЕТ — больше строк не будет без доп. источника урожайности")

if not ndvi_field_years.empty:
    print(f"\nВАРИАНТ 2: Все field×year с NDVI (БЕЗ таргета, для кластеризации)")
    print(f"  Датасет: {len(ndvi_field_years)} строк")
    print("  Но нет таргета → только unsupervised learning / anomaly detection")

yield_not_in_prod = set()
if (
    not yield_maps_df.empty
    and "field_id" in yield_maps_df.columns
    and "year" in yield_maps_df.columns
    and not productivity_df.empty
    and "year" in productivity_df.columns
):
    print(f"\nВАРИАНТ 3: Можем ли получить больше productivity_estimates?")
    yield_maps_field_years = (
        yield_maps_df.groupby(["field_id", "year"])
        .size()
        .reset_index(name="count")
    )
    prod_set = set(zip(productivity_df["field_id"], productivity_df["year"]))
    yield_set = set(zip(yield_maps_field_years["field_id"], yield_maps_field_years["year"]))

    yield_not_in_prod = yield_set - prod_set

    print(f"  Yield_maps field×year: {len(yield_set)}")
    print(f"  Productivity_estimates field×year: {len(prod_set)}")
    print(f"  Yield_maps НЕ в productivity_estimates: {len(yield_not_in_prod)}")

    if len(yield_not_in_prod) > 0:
        print(
            f"  ⚠️ Можем добавить до {len(yield_not_in_prod)} строк, агрегируя урожайность из yield_maps"
        )
        print(f"  Примеры (первые 10):")
        for field_id, year in list(yield_not_in_prod)[:10]:
            print(f"    field_id={field_id}, year={year}")

# ============================================================================
# 8. РЕКОМЕНДАЦИИ
# ============================================================================
print("\n" + "=" * 80)
print("8. РЕКОМЕНДАЦИИ: ЧТО ЕЩЁ ВЫТАЩИТЬ")
print("=" * 80)

print("\n1. ✅ ВЫТАЩИТЬ УРОЖАЙНОСТЬ ИЗ YIELD_MAPS")
if yield_not_in_prod:
    print(
        f"   Yield_maps содержат field×year без productivity_estimates: {len(yield_not_in_prod)} комбинаций"
    )
    print("   Можно агрегировать карты до среднего по полю/году и расширить таргет.")
else:
    print("   Либо yield_maps нет, либо все их годы уже покрыты productivity_estimates.")

print("\n2. 🌦️ ВЫТАЩИТЬ МЕТЕОДАННЫЕ (Virtual Weather Stations / Weather history)")
print("   - Температура, осадки, влажность по field×date")
print("   - Агрегировать по сезону: средняя температура, сумма осадков и т.п.")
print("   - Это критично для учёта годовой вариативности (то, что сейчас ломает CV).")

if not ndvi_df.empty and "year" in ndvi_df.columns:
    print("\n3. 🌱 Диапазон NDVI по годам:")
    print(f"   Сейчас NDVI покрывает годы: {ndvi_df['year'].min()}–{ndvi_df['year'].max()}")
    print("   Можно проверить, есть ли в API более ранние/поздние годы для тех же полей.")

print("\n4. 🚜 Проверить другие эндпоинты API")
print("   - Soil / почвенные параметры (из soil карт)")
print("   - История культур (crop rotation) через history items crops/fields")
print("   - Орошение / техника, если релевантно задаче.")

print("\n" + "=" * 80)
print("ИНВЕНТАРИЗАЦИЯ ЗАВЕРШЕНА")
print("=" * 80)

