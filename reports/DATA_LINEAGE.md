# Паспорт данных (Data Lineage) — Cropwise Yield Prediction

Этот файл фиксирует, **на основе каких входных данных** и **какими скриптами** получены датасеты, метрики, таблицы и рисунки, которые используются в статье и слайдах.

## TL;DR (для преподавателя)

- **Что прогнозируем**: `target_yield_t_ha` (т/га) для пары **поле–год** (`field_id × year`).
- **Главный датасет**: `data_processed/ml_dataset_full_extended.csv` (441 строка = 30 полей × годы 2010–2025).
- **Главные блоки признаков**: NDVI (вегетация), метео (сезонные агрегаты), почва (последний soil test), геометрия, культура (`crop_id`, `prev_crop_id`).
- **Валидация**: `GroupKFold` по `year`, метрики по культурам считаются на **OOF‑предсказаниях**.

## 1) Единица наблюдения и таргет

- **Единица наблюдения**: `field_id × year` (поле–год)
- **Таргет**: `target_yield_t_ha` (т/га), согласованный и приведённый к единому формату на уровне `(field_id, year)`

## 2) Сырые входные данные (raw)

| Файл | Где используется | Что содержит (кратко) |
|---|---|---|
| `data_raw/fields.csv` | `build_ml_dataset.py`, `compare_by_crops_and_cropwise.py` | Справочник полей: `id`, `name`, геометрия, площади, координаты |
| `data_raw/crops.csv` | `build_ml_dataset.py`, `compare_by_crops_and_cropwise.py` | Справочник культур: `id`, `name` (для маппинга `crop_id`) |
| `data_raw/productivity_estimates.csv` | `build_ml_dataset.py` | Основной источник таргета: `field_id`, `year`, `estimate_value` |
| `data_raw/additional_yields_from_history.csv` | `build_ml_dataset.py` | Дополнительные значения урожайности (добавляются только для новых `(field_id, year)`, которых нет в `productivity_estimates`) |
| `data_raw/yield_maps.csv` | `build_ml_dataset.py` | Карты урожайности; используются как дополнительный источник таргета для новых `(field_id, year)` (берётся `calculated_average` или `totals.result.average.value`) |
| `data_raw/history_items_full.csv` | `build_ml_dataset.py` | История культур по полям/годам: используется для `crop_id` и `prev_crop_id` |
| `data_raw/ndvi_timeseries.csv` | `build_ml_dataset.py` | Временные ряды NDVI по полям и датам |
| `data_raw/soil_tests.csv` | `build_ml_dataset.py` | Почвенные показатели по полям (pH, OM, P, K, N, Mg, CEC и т.п.), берётся **последний** SoilTest по `made_at` на поле |
| `data_raw/weather_season_aggregates.csv` | `build_ml_dataset.py` | Сезонные агрегаты погоды по `(field_id, year)` (температуры, осадки, снег) |
| `data_raw/operations.csv` | `build_ml_dataset.py` | Агрооперации: используются для `ops_*`‑фич (в т.ч. NPK из `application_mix_items`, даты посева/уборки) и для датасета `ml_dataset_ops_ndvi_2021_2025.csv` |
| `data_raw/fertilizers_npk.csv` | `build_ml_dataset.py` (опционально) | Справочник NPK состава удобрений: `id`, `N`, `P2O5`, `K2O` (для более точного расчёта `ops_npk_*`) |
| `data_raw/seeds.csv` | `build_ml_dataset.py` (опционально) | Каталог семян: используется для восстановления `crop_id` через `application_mix_items` (seed_id → crop_id), если он есть |
| `data_raw/productivity_data.csv` | `compare_by_crops_and_cropwise.py` | Внешний экспорт “Cropwise productivity”. Важно: столбец “прогноз” **совпадает с таргетом** на пересечении (не независимый прогноз) |

### 2.2 Raw файлы, которые лежат в `data_raw/`, но **не участвуют** в текущей сборке ML‑датасета

Эти файлы сохранены “про запас” (или для будущих итераций), но **в текущем пайплайне** (сборка `ml_dataset_full_extended.csv` + обучение/оценка) они не используются:

- `data_raw/plant_threats.csv`
- `data_raw/field_scout_reports_aggregated.csv`
- `data_raw/growth_stage_groups.csv`
- `data_raw/growth_stages.csv`
- `data_raw/growth_scales.csv`
- `data_raw/productivity_estimate_peers.csv`
- `data_raw/productivity_estimate_histories.csv`
- `data_raw/weather_history_items.csv`
- `data_raw/soil_test_samples.csv`
- `data_raw/field_to_station_mapping.csv`
- `data_raw/weather_stations.csv`

### 2.1 Схемы входных таблиц (ключи и важные поля)

Ниже перечислены **ключи объединения** и **минимальный набор полей**, которые реально используются в пайплайне.

| Источник | Гранулярность | Ключ(и) | Важные поля (пример) |
|---|---|---|---|
| `data_raw/fields.csv` | поле | `id` | `id`, `name`, `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`, (геометрия/контур) |
| `data_raw/crops.csv` | культура (справочник) | `id` | `id`, `name` |
| `data_raw/ndvi_timeseries.csv` | поле–дата (внутри года) | `field_id`, `date` | `field_id`, `date`, `ndvi` |
| `data_raw/weather_season_aggregates.csv` | поле–год | `field_id`, `year` | `temp_*`, `precip_*`, `snow_*` (сезонные агрегаты) |
| `data_raw/soil_tests.csv` | поле–дата теста | `field_id`, `made_at` | `field_id`, `made_at`, `soil_pH.value`, `soil_organic_matter.value`, `soil_P.value`, `soil_K.value`, `soil_N.value`/`soil_N_NO3.value`, `soil_Mg.value`, `soil_cation_exchange_capacity.value`, `soil_Ca_saturation.value` |
| `data_raw/operations.csv` | операция | `field_id`, `performed_at` (и др.) | тип операции (soil/application/harvesting/…), даты, нормы и NPK (если есть) |
| `data_raw/productivity_data.csv` | поле–год (после агрегации) | `Поле`→`field_id`, `Год` | `урожайность факт ц/га`, `урожайность прогноз ц/га` *(в этом экспорте прогноз = таргет)* |

Примечание: файлы `operations.csv` и признаки `ops_*` используются только в отдельной подвыборке `ops_ndvi_2021_2025` и **не являются основой финальной модели**.

## 3) Обработанные датасеты (processed)

| Файл | Кто создаёт | Что это |
|---|---|---|
| `data_processed/ml_dataset_full_extended.csv` | `build_ml_dataset.py` | Основной датасет `field_id×year`: геометрия + культура + NDVI + метео + почва + `target_yield_t_ha` |
| `data_processed/ml_dataset_ndvi_only_extended.csv` | `build_ml_dataset.py` | Вариант с расширенными NDVI + культура/геометрия (+ часть метео/почвы по текущей сборке) |
| `data_processed/ml_dataset_with_ndvi.csv` | `build_ml_dataset.py` | Базовый датасет с NDVI (историческая конфигурация) |
| `data_processed/ml_dataset_ops_ndvi_2021_2025.csv` | `build_ml_dataset.py` | Подвыборка 2021–2025 с детальными агрооперациями + NDVI |

### 3.1 Что именно попадает в `ml_dataset_full_extended.csv`

`data_processed/ml_dataset_full_extended.csv` содержит (минимальный состав):

- **Ключи**: `field_id`, `year`
- **Таргет**: `target_yield_t_ha`
- **Культура**: `crop_id`, `prev_crop_id`
- **Геометрия**: `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long` (и др. `field_*`)
- **NDVI**: `ndvi_*` (агрегаты по сезону и фазам)
- **Почва**: `field_soil_*` (последний тест на поле)
- **Метео**: `weather_*` (сезонные агрегаты)

## 4) Обучение моделей и метрики

### 4.1 Baseline и Iteration 1 (Pipeline + OOF)

| Артефакт | Кто создаёт | Что содержит |
|---|---|---|
| `models/models_comparison.csv` | `train_baseline_model.py` | Метрики (CV/Train) по всем датасетам и наборам признаков |
| `models/oof_predictions_full_extended_all_no_leak.csv` | `train_baseline_model.py` | OOF‑предсказания baseline‑модели для `full_extended + all_no_leak` |
| `reports/tables/models_comparison_summary.csv` | `generate_final_report.py` | Сжатая таблица результатов для отчёта/статьи |
| `reports/figures/cv_r2_comparison.png` | `generate_final_report.py` | Сравнение CV R² по датасетам/сценариям |
| `reports/final_report.pdf` | `generate_final_report.py` | Финальный PDF‑отчёт (сводные таблицы и рисунки) |

**Как считаются “честные” метрики**:
- Используется `GroupKFold` по `year`.
- Заполнение пропусков делается **внутри** `sklearn.Pipeline` (SimpleImputer), чтобы не было leakage.
- Для анализа по культурам/группам используются **OOF‑предсказания**.

### 4.2 Iteration 2 (выбранная финальная модель)

#### Лучший регуляризованный GBDT (финальная модель)

- Параметры: `max_depth=3`, `min_samples_leaf=5`, `subsample=0.7`, `learning_rate=0.05`, `n_estimators=300`

| Артефакт | Кто создаёт | Что содержит |
|---|---|---|
| `models/models_iteration2_gbdt_full_extended.csv` | `train_baseline_model.py` | Grid search по гиперпараметрам GBDT для `full_extended + all_no_leak` |
| `models/oof_predictions_full_extended_all_no_leak_reg.csv` | `train_iteration2_best_gbdt_full_extended.py` | OOF‑предсказания **финальной** модели (Regularized GBDT) |
| `models/predictions_plot_full_extended_all_no_leak_reg_oof.png` | `train_iteration2_best_gbdt_full_extended.py` | Predictions vs Actual на OOF |
| `models/feature_importance_full_extended_all_no_leak_reg.png` | `train_iteration2_best_gbdt_full_extended.py` | Важности признаков (impurity‑based `feature_importances_`) |

#### CatBoost (проверка альтернативного алгоритма)

| Артефакт | Кто создаёт | Что содержит |
|---|---|---|
| `models/models_iteration2_catboost_full_extended.csv` | `train_iteration2_catboost.py` | Метрики CatBoost на `full_extended + all_no_leak` (несколько конфигураций) |

#### Финальная таблица “3 модели”

| Артефакт | Кто создаёт | Что содержит |
|---|---|---|
| `reports/tables/models_comparison_3models.csv` | вручную/сборка итогов | Baseline GBDT vs CatBoost vs Regularized GBDT ★ (CV/Train метрики) |

## 5) Метрики по культурам и группам (OOF)

Скрипт: `compare_by_crops_and_cropwise.py`

Он строит общий датасет сравнения и считает метрики по культурам/группам, используя **OOF‑предсказания** (приоритетно — финальная модель `oof_reg`).

| Артефакт | Кто создаёт | Что содержит |
|---|---|---|
| `data_processed/comparison_model_vs_cropwise.csv` | `compare_by_crops_and_cropwise.py` | Таблица `(field_id, year)` с таргетом, предсказанием модели и (где есть) “Cropwise” |
| `reports/tables/metrics_by_crop_group_model_vs_cropwise.csv` | `compare_by_crops_and_cropwise.py` | Метрики по группам культур (grain/oilseeds/…) |
| `reports/tables/metrics_by_crop_model_vs_cropwise.csv` | `compare_by_crops_and_cropwise.py` | Метрики по отдельным культурам |
| `reports/tables/draft_section_5_crops_comparison.md` | `compare_by_crops_and_cropwise.py` | Черновой текст для подпункта статьи по культурам |

**Важное примечание про Cropwise**:
- В текущем `data_raw/productivity_data.csv` столбец “прогноз Cropwise” на пересекающихся `(field_id, year)` совпадает с таргетом (R²=1, RMSE=0).  
  Поэтому сравнение “наша модель vs Cropwise” **не интерпретируется** как сравнение двух независимых моделей.

## 6) Презентация/слайды

| Файл | Что использует внутри |
|---|---|
| `Прогноз Урожайности Cropwise.html` | Вставляет картинки из `models/` и таблицы/метрики, соответствующие Regularized GBDT (OOF) |

## 7) Как воспроизвести результаты (команды)

Запуск из корня проекта:

1) Сборка датасетов:
- `python build_ml_dataset.py`

2) Обучение baseline‑моделей + первичный отчёт:
- `python train_baseline_model.py`

3) Финальная модель (Regularized GBDT) + OOF/графики:
- `python train_iteration2_best_gbdt_full_extended.py`

4) Метрики по культурам/группам (OOF финальной модели):
- `python compare_by_crops_and_cropwise.py`


