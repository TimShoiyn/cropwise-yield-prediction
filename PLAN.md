# Cropwise Yield Prediction — План работ (PhD)

> Цель: довести проект до уровня Q1/Q2 публикации в agronomy/remote sensing журнале.
> Главный научный вклад: **as-of yield forecast** — ML-модель прогнозирует урожайность на 1-2 месяца раньше уборки лучше/сравнимо с production-системой Cropwise.

---

## 0. Контекст и диагноз (что нашли при аудите)

### 0.1 Объект
- **N = 441** наблюдение `(field_id, year)`, **2010–2025**.
- **Одно хозяйство** (29 полей в `field_group_id=1`, +1 в group=6). Радиус ~5 км.
- Доминирует подсолнечник: 286 / 441 = **65%**. Пшеница (яр+оз) = 24%. Остальные культуры — единицы.

### 0.2 Критичные проблемы (отсортированы по импакту)

| # | Проблема | Импакт |
|---|---|---|
| 1 | **Хаос единиц в таргете**. `productivity` смешивает ц/га и т/га в одной колонке. Доказано через `productivity / (harvested_weight/area) ≈ 10` для зерновых, но `≈ 1` для масличных в части записей | **Blocker**: все метрики невалидны до фикса |
| 2 | Все 441 строк из 1 хозяйства, `lat/long` std≈0.04°, `field_group_id` почти константа | High |
| 3 | `field_soil_CEC` = 100% NaN, `Ca_saturation` = 100%, `N_NO3` = 99% | High |
| 4 | Soil-фичи — одно измерение на поле, константа во времени. Не объясняет межгодовую вариацию | Medium |
| 5 | `ops_*` (NPK, даты сева/уборки) есть только за **2021–2025** | High |
| 6 | Phase-features `ndvi_wheat_*` 85% NaN, `ndvi_sunflower_p3` 66% NaN | Medium |
| 7 | Класс-имбаланс по культурам: подсолнечник 65%, остальное «шум» | High |
| 8 | NDVI агрегаты «за весь сезон» включают NDVI до сева и после уборки | Medium |
| 9 | `crop_id`, `prev_crop_id` идут как числовые в GBDT (некорректно) | High |
| 10 | GroupKFold по году ≠ Walk-Forward (модель видит будущее) | Medium |
| 11 | Train R²=0.99 при CV R²=0.31 — переобучение на 441 точку | Medium |
| 12 | NDVI peak/late до конца сезона = leakage для оперативного прогноза | High |
| 13 | `productivity_estimate_histories.csv` (4774 еженедельных прогноза Cropwise) почти не используется как benchmark | High (упущенный научный вклад) |

### 0.3 Агрономические замечания
- GDD считается с 1 января независимо от культуры → для озимой пшеницы бессмысленно.
- Tbase для подсолнечника 10°C — устаревшее значение (правильно 6–8°C для современных гибридов).
- Нет водного баланса в критические фазы (колошение–налив для пшеницы, цветение для подсолнечника).
- Севооборот используется как `prev_crop_id` (число), но реальный сигнал — это пара `(prev_crop, current_crop)` и счётчик «лет после подсолнечника».

---

## 1. Архитектура решения (целевая)

```
data_raw/                                  (без изменений — сырые экспорты)
├── productivity_estimates.csv             # estimate_value (прогноз Cropwise, 440 строк)
├── productivity_estimate_histories.csv    # 4774 еженедельных Cropwise прогноза (золото для as-of)
├── history_items_full.csv                 # productivity, harvested_weight, sowing/harvest dates
├── ndvi_timeseries.csv                    # 757k наблюдений NDVI
├── weather_history_items.csv              # суточная погода
├── operations.csv                         # 2021+ операции
├── fields.csv, crops.csv, seeds.csv       # справочники

data_processed/
├── targets_cleaned_t_ha.csv               # ★ NEW: чистый таргет в т/га (унифицированные единицы)
├── targets_cleaned_audit.csv              # ★ NEW: лог конвертации (что/как/почему)
├── ml_dataset_clean_v2.csv                # ★ NEW: финальный ML датасет (без мусорных фич)
└── ml_dataset_clean_v2_asof_{date}.csv    # ★ NEW: as-of cuts по датам прогноза

models_v2/                                  # ★ NEW: артефакты новых моделей
├── catboost_clean_allcrops/
├── catboost_clean_sunflower/
├── catboost_clean_wheat/
└── asof_comparison/                       # ML vs Cropwise по датам

reports/
├── clean_dataset_audit.md                 # ★ Sprint 1.3
├── model_comparison_v1_vs_v2.md           # ★ Sprint 2.3
├── asof_forecast_results.md               # ★ Sprint 3.1 (главный артефакт для статьи)
└── FINAL_RESULTS.md                       # ★ Sprint 4
```

---

## 2. Спринт 1 — Очистка таргета и данных (1–2 дня)

### 2.1 `scripts/clean/build_clean_targets.py`
**Цель**: получить единый таргет `target_yield_t_ha` в честных т/га для всех `(field_id, year)`.

**Источники (по приоритету)**:
1. `harvested_weight / tillable_area` из `history_items_full.csv` — самый надёжный физический таргет (~129 строк).
2. `productivity` из `history_items_full.csv` — после нормализации единиц.
3. `estimate_value` из `productivity_estimates.csv` — прогноз Cropwise, после нормализации (440 строк).

**Правила нормализации** (per-crop, на основе агрономических потолков):

| standard_name | Реалистичный max т/га (P99) | Правило unit-fix |
|---|---|---|
| sunflower | 5.0 | если > 5 → делить на 10 (это ц/га) |
| wheat_spring | 7.0 | если > 7 → делить на 10 |
| wheat_winter | 8.0 | если > 8 → делить на 10 |
| barley_spring | 7.0 | если > 7 → делить на 10 |
| maize | 12.0 | если > 12 → делить на 10 |
| oil_seed_raps_spring | 4.5 | если > 4.5 → делить на 10 |
| oil_seed_raps_winter | 5.0 | если > 5 → делить на 10 |
| soya | 4.0 | если > 4 → делить на 10 |
| pea | 4.5 | если > 4.5 → делить на 10 |
| другие | 5.0 (default) | если > 5 → делить на 10 |

**Outlier removal**: после нормализации удалять записи где `target = 0` или `target < P5 культуры` или `target > P99.5 культуры`.

**Артефакты**:
- `data_processed/targets_cleaned_t_ha.csv`: `field_id, year, crop_id, target_yield_t_ha, target_source, unit_fix_applied, kept`
- `reports/target_cleaning_audit.md`: разбивка по культурам — сколько строк, сколько fix-нуто, сколько выкинуто, до/после распределения

### 2.2 `scripts/clean/build_clean_dataset.py`
**Цель**: пересобрать ML-датасет на чистом таргете и выкинуть мусорные фичи.

**Действия**:
1. Берём `data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv` как базу.
2. Заменяем `target_yield_t_ha` на чистый из `targets_cleaned_t_ha.csv` (inner join).
3. **Удаляем фичи**:
   - `field_lat`, `field_long` (constant per farm)
   - `field_group_id` (almost constant)
   - `field_soil_CEC`, `field_soil_Ca_saturation`, `field_soil_N_NO3` (≥99% NaN)
   - все harvest-leakage: `ops_count_harvesting`, `ops_days_seeding_to_harvest`, `ops_season_end`, `ops_season_duration_days`, `ops_yield_t_ha`
4. **Оставляем**: NDVI (mean/max/early/mid + integral), weather (precip, hot_days, gdd), soil (pH/OM/P/K/N/Mg), `crop_id`, `prev_crop_id`, `field_id`, `field_tillable_area`, `year`.
5. **Кодировка категорий**: ничего не one-hot-им — оставляем как Int для CatBoost.

**Артефакт**: `data_processed/ml_dataset_clean_v2.csv`

### 2.3 EDA-отчёт `reports/clean_dataset_audit.md`
- shape до/после
- target distribution per crop (boxplot)
- per-crop NaN ratios
- correlation matrix top-30 фич × target (по культуре)
- год × число строк
- conclusion: какие фичи имеют сигнал, какие можно дропнуть на след спринтах

---

## 3. Спринт 2 — Честный baseline на чистых данных (2 дня)

### 3.1 `scripts/train/train_clean_catboost.py`
- **CatBoostRegressor** с `cat_features=['crop_id', 'prev_crop_id', 'field_id']`
- **Walk-Forward CV**: для каждого test_year ∈ [2017..2025] тренировка на годах ≤ test_year-1 (мин 5 лет в трейне)
- Сохраняем OOF предсказания, fold metrics, feature importance, SHAP top-20
- Метрики: R², RMSE (т/га), MAE (т/га), MAPE (%) — теперь они валидны

### 3.2 Per-crop модели
- `sunflower` (n≈280 после чистки)
- `wheat_combined` (wheat_spring + wheat_winter, n≈100)
- Без других культур (n<20 для каждой — нет смысла)

### 3.3 Сравнение
- `reports/model_comparison_v1_vs_v2.md`:
  - старая `models_phases_comparison.csv` vs новые CatBoost метрики
  - на чистом таргете старая модель тоже переоценена (для честности)
  - ожидание: R² немного упадёт (потому что таргет стал «честнее»), но MAPE станет интерпретируемым

---

## 4. Спринт 3 — Главный научный вклад (2–3 дня)

### 4.1 As-of forecast experiment (`scripts/asof/build_asof_dataset.py` + `train_asof.py`)

**Идея**: для каждой даты прогноза D ∈ {1 июля, 1 августа, 1 сентября} построить датасет, где **все time-series фичи (NDVI, weather) обрезаны на D**:
- `ndvi_*_asof_D` = агрегаты NDVI за период [сев, D]
- `weather_*_asof_D` = аналогично
- soil/crop/field — без изменений

Обучить ту же CatBoost-модель (per-crop) для каждой даты D.

**Бенчмарк Cropwise**: из `productivity_estimate_histories.csv` для каждой `(field_id, year, D)` достать ближайший Cropwise прогноз (по `estimate_history` JSON). Для дат D Cropwise тоже даёт прогноз → можем напрямую сравнить.

### 4.2 Главный график статьи
- Ось X: дата прогноза (1 июля, 1 авг, 1 сент, post-harvest)
- Ось Y: MAPE (%)
- Линии: ML model (sunflower), ML model (wheat), Cropwise (sunflower), Cropwise (wheat)
- Идея: ML на 1 июля точнее Cropwise (раннее обнаружение проблемных полей), к 1 сентября оба сходятся

### 4.3 Агрономические фичи v2
- GDD от даты сева (sowing_date из ops_*) с правильными Tbase: пшеница 5°C, подсолнечник 6°C
- DSI (Drought Stress Index) и HSI (Heat Stress Index) **в критическую фазу налива** для пшеницы / **цветения** для подсолнечника
- `crop_pair = (prev_crop, current_crop)` как cat_feature
- `years_since_sunflower` (для поля) — счётчик заразихи

---

## 5. Спринт 4 — Финал и статья (1 день)

### 5.1 `reports/FINAL_RESULTS.md`
- Сводная таблица всех моделей (v1 → v2 → v2+asof → v2+agro)
- 3-4 главных графика для статьи
- Сравнение с Cropwise
- Честное обсуждение ограничений (1 хозяйство, малая выборка)

### 5.2 Что писать в статье
- **НЕ** «у нас R²=0.998». Реалистично R²≈0.30–0.45 для подсолнечника, MAPE 13–18%.
- **ДА** «ML догоняет/обгоняет production-Cropwise на ранних датах прогноза, что даёт 1–2 месяца форы для агрономических решений».
- Ограничения: 1 хозяйство, продолжение требует мультирегиональной выборки.

---

## 6. Что НЕ делаем сейчас (out of scope)

- Глубокие сети (LSTM/Transformer на NDVI time-series) — нет смысла на N=441
- Дозагрузка новых полей через API — это политическое решение заказчика
- EVI/NDRE — нужен другой API доступ
- Soil temperature/moisture с датчиков — нет в наличии

---

## 7. Сроки

| Спринт | Дни | Главный артефакт |
|---|---|---|
| 1 | 1–2 | Чистый таргет + датасет |
| 2 | 2 | CatBoost baseline + per-crop |
| 3 | 2–3 | As-of forecast (ГЛАВНОЕ) |
| 4 | 1 | Финальный отчёт |

**Итого: 6–8 рабочих дней**.
