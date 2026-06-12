"""
Диагностика operations.csv для извлечения crop_id и seeding информации.

Запуск:
  python diagnostic_operations.py
"""

from __future__ import annotations

import ast
from collections import Counter

import pandas as pd


OPS_CSV = "data_raw/operations.csv"
PRODUCTIVITY_CSV = "data_raw/productivity_estimates.csv"
CROPS_CSV = "data_raw/crops.csv"

OUT_SEED_CSV = "diagnostic_seed_operations.csv"


def _safe_parse_mix_items(raw: object) -> list[dict]:
    """
    В твоём operations.csv поле application_mix_items — это строка с Python-литералом вида:
    "[{'applicable_id': 10, 'applicable_type': 'Fertilizer', ...}]"
    Это НЕ JSON, поэтому json.loads() не работает. Используем ast.literal_eval().
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s or s.lower() == "nan":
        return []
    try:
        items = ast.literal_eval(s)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return []
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]
    return []


def main() -> None:
    # Загрузить данные
    operations_df = pd.read_csv(OPS_CSV)

    print("=" * 80)
    print("ДИАГНОСТИКА OPERATIONS.CSV")
    print("=" * 80)
    print(f"Файл: {OPS_CSV}")
    print(f"Строк: {len(operations_df)}")

    # 1. Проверить наличие crop_id на уровне операций
    print("\n1. CROP_ID В ОПЕРАЦИЯХ:")
    if "crop_id" in operations_df.columns:
        print("   ✅ Колонка crop_id существует")
        print(f"   Заполнено: {operations_df['crop_id'].notna().sum()}/{len(operations_df)}")
        print(f"   Уникальных crop_id: {operations_df['crop_id'].nunique(dropna=True)}")
        print(f"   Примеры crop_id:\n{operations_df['crop_id'].value_counts(dropna=True).head(10)}")
    else:
        print("   ❌ Колонка crop_id отсутствует")

    # 2. Проверить operation_type для seeding
    print("\n2. ОПЕРАЦИИ ТИПА SEEDING:")
    if "operation_type" not in operations_df.columns:
        print("   ❌ Колонка operation_type отсутствует")
        seeding_ops = operations_df.iloc[0:0]
    else:
        seeding_ops = operations_df[operations_df["operation_type"] == "seeding"].copy()
        print(f"   Всего операций seeding: {len(seeding_ops)}")
        if len(seeding_ops) > 0:
            print(f"   Уникальных field_id: {seeding_ops['field_id'].nunique()}")
            if "season" in seeding_ops.columns:
                print(f"   Распределение по сезонам:\n{seeding_ops['season'].value_counts().sort_index()}")
            cols = [c for c in ["field_id", "season", "operation_type", "actual_start_datetime", "completed_date"] if c in seeding_ops.columns]
            print("\n   Пример операции seeding:")
            print(seeding_ops[cols].head(3).to_string(index=False))

    # 3. Проверить application_mix_items для Seed
    print("\n3. SEED В APPLICATION_MIX_ITEMS:")
    if "application_mix_items" not in operations_df.columns:
        print("   ❌ Колонка application_mix_items отсутствует")
        seed_df = pd.DataFrame()
    else:
        seed_operations: list[dict] = []
        seed_applicable_ids: list[int] = []

        non_null = operations_df[operations_df["application_mix_items"].notna()]
        for _, row in non_null.iterrows():
            items = _safe_parse_mix_items(row["application_mix_items"])
            if not items:
                continue
            for item in items:
                if item.get("applicable_type") == "Seed":
                    seed_operations.append(
                        {
                            "field_id": row.get("field_id"),
                            "season": row.get("season"),
                            "operation_type": row.get("operation_type"),
                            "agro_operation_id": row.get("id"),
                            "mix_item_applicable_id": item.get("applicable_id"),
                            "mix_item_applicable_name": item.get("applicable_name"),
                            "mix_item_applicable_type": item.get("applicable_type"),
                            "mix_item_fact_rate": item.get("fact_rate"),
                            "mix_item_planned_rate": item.get("planned_rate"),
                            "mix_item_rate_unit": item.get("rate_unit"),
                            "mix_item_unit_id": item.get("unit_id"),
                            "actual_start_datetime": row.get("actual_start_datetime"),
                            "completed_date": row.get("completed_date"),
                        }
                    )
                    if item.get("applicable_id") is not None:
                        try:
                            seed_applicable_ids.append(int(item.get("applicable_id")))
                        except Exception:
                            pass

        seed_df = pd.DataFrame(seed_operations)
        print(f"   Найдено операций с Seed: {len(seed_df)}")

        if len(seed_df) > 0:
            print(f"   Уникальных field_id: {seed_df['field_id'].nunique()}")
            print(f"   Уникальных applicable_id: {seed_df['mix_item_applicable_id'].nunique(dropna=True)}")
            print("\n   Топ-10 Seed по applicable_id:")
            print(seed_df["mix_item_applicable_id"].value_counts(dropna=True).head(10))

            show_cols = [
                c
                for c in [
                    "field_id",
                    "season",
                    "operation_type",
                    "mix_item_applicable_id",
                    "mix_item_applicable_name",
                    "mix_item_fact_rate",
                    "mix_item_rate_unit",
                    "actual_start_datetime",
                    "completed_date",
                ]
                if c in seed_df.columns
            ]
            print("\n   Примеры Seed записей:")
            print(seed_df[show_cols].head(10).to_string(index=False))

            # Сохранить для дальнейшего использования
            seed_df.to_csv(OUT_SEED_CSV, index=False)
            print(f"\n   💾 Сохранено в {OUT_SEED_CSV}")

            if seed_applicable_ids:
                top = Counter(seed_applicable_ids).most_common(10)
                print("\n   Топ-10 Seed applicable_id (из mix_items, raw):")
                for k, v in top:
                    print(f"     - {k}: {v}")

    # 4. Проверить связь field×year с productivity_estimates
    print("\n4. ПОКРЫТИЕ FIELD×YEAR С CROP/SEED ИНФОРМАЦИЕЙ:")
    productivity_df = pd.read_csv(PRODUCTIVITY_CSV)
    field_years = set(zip(productivity_df["field_id"].astype(int), productivity_df["year"].astype(int)))

    # Из seeding операций (operation_type==seeding)
    seeding_coverage: set[tuple[int, int]] = set()
    if len(seeding_ops) > 0 and "field_id" in seeding_ops.columns and "season" in seeding_ops.columns:
        for _, r in seeding_ops.iterrows():
            try:
                seeding_coverage.add((int(r["field_id"]), int(r["season"])))
            except Exception:
                pass

    # Из Seed в mix_items
    seed_coverage: set[tuple[int, int]] = set()
    if isinstance(seed_df, pd.DataFrame) and len(seed_df) > 0:
        for _, r in seed_df.iterrows():
            try:
                seed_coverage.add((int(r["field_id"]), int(r["season"])))
            except Exception:
                pass

    both = seeding_coverage | seed_coverage
    print(f"   Всего field×year в productivity_estimates: {len(field_years)}")
    print(f"   Покрыто seeding операциями: {len(field_years & seeding_coverage)} ({len(field_years & seeding_coverage)/len(field_years)*100:.1f}%)")
    print(f"   Покрыто Seed в mix_items: {len(field_years & seed_coverage)} ({len(field_years & seed_coverage)/len(field_years)*100:.1f}%)")
    print(f"   Покрыто любым способом: {len(field_years & both)} ({len(field_years & both)/len(field_years)*100:.1f}%)")

    # 5. Проверить crops.csv
    print("\n5. CROPS.CSV СПРАВОЧНИК:")
    crops_df = pd.read_csv(CROPS_CSV)
    print(f"   Файл: {CROPS_CSV}")
    print(f"   Всего культур в справочнике: {len(crops_df)}")
    print(f"   Колонки: {crops_df.columns.tolist()}")

    show_cols = [c for c in ["id", "name", "standard_name", "season_type"] if c in crops_df.columns]
    if show_cols:
        print("\n   Пример культур (head):")
        print(crops_df[show_cols].head(10).to_string(index=False))

    # 6. Проверить applicable_id из Seed vs crops.id
    if isinstance(seed_df, pd.DataFrame) and len(seed_df) > 0 and "mix_item_applicable_id" in seed_df.columns and "id" in crops_df.columns:
        seed_ids = set(seed_df["mix_item_applicable_id"].dropna().astype(int))
        crop_ids = set(crops_df["id"].dropna().astype(int))
        matching = seed_ids & crop_ids

        print("\n6. СОВПАДЕНИЕ SEED APPLICABLE_ID С CROPS.ID:")
        print(f"   Уникальных Seed applicable_id: {len(seed_ids)}")
        print(f"   Уникальных crop_id в справочнике: {len(crop_ids)}")
        if seed_ids:
            print(f"   Совпадений: {len(matching)} ({len(matching)/len(seed_ids)*100:.1f}%)")
        else:
            print("   Совпадений: 0 (seed_ids пуст)")

        if len(matching) > 0:
            print("   ✅ Seed applicable_id можно использовать как crop_id (по совпадению id)!")
            matching_crops = crops_df[crops_df["id"].isin(matching)].copy()
            if show_cols:
                print("\n   Примеры совпадающих crop:")
                print(matching_crops[show_cols].head(10).to_string(index=False))
        else:
            print("   ⚠️  Seed applicable_id НЕ совпадают с crops.id — нужен другой справочник/эндпоинт Seeds.")

    print("\n" + "=" * 80)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

