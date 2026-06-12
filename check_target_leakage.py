"""
Проверка утечек таргета в ml_dataset_full_extended.csv:
- выводим список колонок;
- проверяем, не совпадает ли какая‑то колонка по значениям с target_yield_t_ha;
- считаем корреляции target_yield_t_ha со всеми числовыми фичами.
"""

import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_processed" / "ml_dataset_full_extended.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("=== Файл:", DATA_PATH)
    print("Строк:", len(df), "Колонок:", len(df.columns))
    print("\nСписок колонок:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:3d}. {col}")

    target = "target_yield_t_ha"
    if target not in df.columns:
        print(f"\n[ERROR] Колонка {target!r} не найдена в датасете.")
        return

    tgt = pd.to_numeric(df[target], errors="coerce")

    equal_cols: list[str] = []
    print("\n=== Проверка совпадения колонок с target_yield_t_ha ===")
    for col in df.columns:
        if col == target:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.equals(tgt):
            equal_cols.append(col)

    if equal_cols:
        print("Найдены колонки, идентичные таргету по значениям:", equal_cols)
    else:
        print("Нет ни одной колонки, полностью совпадающей с таргетом.")

    print("\n=== Корреляции c target_yield_t_ha (top 20) ===")
    num_df = df.select_dtypes(include=[np.number])
    corrs = num_df.corr()[target].sort_values(ascending=False)
    print(corrs.head(20))


if __name__ == "__main__":
    main()

