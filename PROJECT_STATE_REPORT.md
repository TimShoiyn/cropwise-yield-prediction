## Общее состояние проекта (на сейчас)

### 1. Что за данные мы собрали

- **Основной объект прогноза**
  - **Таргет**: `target_yield_t_ha` (т/га) по парам `(field_id, year)`.
  - **Базовый таргет**: `data_raw/productivity_estimates.csv` — ~440 пар `field_id×year` (2010–2024/25, по факту 2010–2024, 2 хозяйства).
  - **Расширенный таргет**: `data_processed/targets_extended.csv` — 440 базовых + ≈1900 дополнительных из `history_items_full` → всего ≈2300 записей, но в ML‑датасеты сейчас попадает 441 уникальная пара `field_id×year` (те, у кого есть NDVI и геометрия).

- **Геометрия полей**
  - **`data_raw/fields.csv`**
    - Колонки: `id`, `name`, `shape_simplified_geojson`, `calculated_area`, `tillable_area`, `lat`, `long`, административные признаки, `created_at`, `updated_at`.
    - Покрытие: ~30 полей (`id` 195–234 и др.).
    - Геометрия в формате `MultiPolygon` (GeoJSON‑строка), плюс агрегированные площади (га) и центр поля (`lat/long`).

- **Культуры / севообороты**
  - **`data_raw/crops.csv`**
    - Справочник культур: `id`, `name`, `season_type` (`spring`/`winter`), `base_crop_id`, служебные поля.
  - **`data_raw/history_items_full.csv`**
    - Полная история полей: для каждого `field_id` и `year` — `crop_id`, `productivity`, даты сева/уборки, прочие атрибуты.
    - Используется для:
      - расширения таргета (`additional_yields_from_history.csv`);
      - восстановления `crop_id` и `prev_crop_id` (культура‑предшественник).
  - **`data_raw/seeds.csv`**
    - Каталог семян (из `/seeds/{id}`), связывает `seed_id` из операций с `crop_id`.

- **Агрооперации и удобрения**
  - **`data_raw/operations.csv`**
    - Все агрооперации по полям и сезонам: `field_id`, `season` (год), `operation_type` (`seeding`, `tillage`, `harvesting`, `application`, `spraying` и т.д.), `actual_start_datetime`, `completed_date`, `application_mix_items` (строка с составом смеси).
    - Используется для:
      - счётчиков операций по типам (`ops_count_*`);
      - дат начала/конца сезона (`ops_season_start/end`, `ops_season_duration_days`);
      - интервалов посев–уборка (`ops_days_seeding_to_harvest`);
      - извлечения доз NPK по `application_mix_items` + `fertilizers_npk`.
  - **`data_raw/fertilizers_npk.csv`**
    - Результат `/fertilizers`: `id`, `name`, `manufacturer_name`, `fertilizer_type`, и поля `N`, `P2O5`, `K2O`, ... (проценты по массе).
    - Маппится по `applicable_id` из `application_mix_items` для вычисления:
      - `ops_npk_n_total`, `ops_npk_p_total`, `ops_npk_k_total`, `ops_npk_total` (кг д.в. на гектар за сезон).

- **NDVI / спутниковые данные**
  - **`data_raw/ndvi_timeseries.csv`**
    - Временные ряды NDVI: `field_id`, `date`, `year`, `ndvi` (или `value`/`index_value` после нормализации), `product_type='ndvi'`.
    - Покрытие:
      - Годы: **2010–2025** (по логам: мин/макс `date` в этом диапазоне).
      - Поля: тот же пул (~30 полей), но не все поля имеют NDVI каждый год.
      - Среднее число наблюдений на `field×year`: 80–120 точек.
  - Из этого считаются агрегаты:
    - `ndvi_mean_season`, `ndvi_max_season`, `ndvi_min_season`, `ndvi_std_season`;
    - фазные средние: `ndvi_early`, `ndvi_mid`, `ndvi_late` (по датам внутри года);
    - `ndvi_trend` (тенденция по времени), `ndvi_observations`, `ndvi_peak_date`, `ndvi_peak_doy`;
    - `ndvi_integral` (интеграл кривой NDVI за сезон), `ndvi_green_period_days`.

- **Погода / станции (пока без истории погоды)**
  - **`data_raw/weather_stations.csv`**
    - Список виртуальных метеостанций: `id`, `latitude`, `longitude`, другие поля.
  - **`data_raw/field_to_station_mapping.csv`**
    - Для каждого поля — ближайшая метеостанция (`field_id`, `station_id`, `distance_km`).
  - **Погодные ряды (`weather_history.csv`) не удалось выкачать** (404 по всем вариантам эндпоинтов).

- **Yield maps**
  - **`data_raw/yield_maps.csv`**
    - Метаданные по картам урожайности (field‑level), по годам ~2014–2024.
    - Использовано только в диагностике для оценки, можно ли расширить таргет через yield maps (пока не интегрировано).

### 2. Обработанные датасеты (ML input) и их структура

- **`data_processed/ml_dataset_with_ndvi.csv`** (**base_with_ndvi**)
  - Размер: **440 строк, 40 колонок**.
  - Строка = `(field_id, year)` с таргетом только из `productivity_estimates`.
  - Фичи:
    - `field_*`: `field_id`, `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`.
    - `crop_id`, `crop_name`, `crop_season_type`.
    - `ops_*`:
      - `ops_count_total`, `ops_count_soil`, `ops_count_application`, `ops_count_harvesting`, `ops_count_seeding`, `ops_count_tillage`, `ops_count_spraying`.
      - `ops_season_start`, `ops_season_end`, `ops_season_duration_days`.
      - `ops_days_seeding_to_harvest`, `ops_days_seeding_to_ndvi_peak`.
      - `ops_yield_t_ha` (сезонная урожайность из операций, потенциальный leakage).
      - `ops_npk_n_total`, `ops_npk_p_total`, `ops_npk_k_total`, `ops_npk_total`.
    - `ndvi_*` как выше (агрегаты по году).

- **`data_processed/targets_extended.csv`**
  - Размер: ≈**2340 строк** (440 + доп. из history), колонки:
    - `field_id`, `year`, `yield_t_ha` (таргет).

- **`data_processed/ml_dataset_ndvi_only_extended.csv`** (**ndvi_only_extended**)
  - Размер: **441 строк, 23 колонки**.
  - Строка = `(field_id, year)` из `targets_extended` **там, где есть NDVI и геометрия**.
  - Фичи:
    - `year`, `target_yield_t_ha`.
    - `crop_id` (текущая культура), `prev_crop_id` (предшественник по `year-1` из history_items).
    - `field_id`, `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`.
    - NDVI‑агрегаты: `ndvi_observations`, `ndvi_mean_season`, `ndvi_max_season`, `ndvi_min_season`, `ndvi_std_season`,
      `ndvi_early`, `ndvi_mid`, `ndvi_late`, `ndvi_trend`, `ndvi_green_period_days`, `ndvi_peak_date`, `ndvi_peak_doy`, `ndvi_integral`.

- **`data_processed/ml_dataset_ops_ndvi_2021_2025.csv`** (**ops_ndvi_2021_2025**)
  - Размер: **136 строк, 41 колонка**.
  - Строка = `(field_id, year)` из `targets_extended`, **фильтр по `ops_count_total > 0`** (по факту годы 2021–2025).
  - Фичи:
    - `field_*`, `year`, `crop_id`, `crop_name`, `crop_season_type`, `prev_crop_id`.
    - `ops_*` (как в базовом датасете, но только там, где реально были операции).
    - NDVI‑агрегаты (аналогично предыдущему датасету).
    - `target_yield_t_ha` (расширенный таргет).

### 3. Модели и их качество (сводка по `models/models_comparison.csv`)

- **Датасет `base_with_ndvi`** (`ml_dataset_with_ndvi.csv`)
  - Лучший по CV: **`features_set = all`** (все фичи, включая NDVI и ops, без выкидывания leakage):
    - CV: `R² ≈ 0.05`, `RMSE ≈ 6.20 т/га`, `MAE ≈ 4.64`, `MAPE ≈ 21.1%`.
    - Train: `R² ≈ 0.99`, `RMSE ≈ 0.79`, `MAPE ≈ 2.87%` → сильный оверфит (train почти идеален, но CV слабый).
  - Вариант `all_no_leak` (без `ops_yield_t_ha`):
    - CV очень похож: `R² ≈ 0.046`, `RMSE ≈ 6.23`, `MAPE ≈ 21.3%`.
    - Train ещё лучше (`R² ≈ 0.991`, `RMSE ≈ 0.72`).

- **Датасет `ndvi_only_extended`** (`ml_dataset_ndvi_only_extended.csv`) — **основной честный baseline**
  - **no_ndvi / no_ndvi_no_leak** (только геометрия + `crop_id`/`prev_crop_id`):
    - CV: `R² ≈ -1.02`, `RMSE ≈ 8.9`, `MAE ≈ 6.64`, `MAPE ≈ 30%` → почти нет сигнала без NDVI.
    - Train: `R² ≈ 0.716`, `RMSE ≈ 4.12`.
  - **ndvi_only** (только NDVI‑фичи):
    - CV: `R² ≈ -0.10`, `RMSE ≈ 6.79`, `MAE ≈ 5.07`, `MAPE ≈ 22.8%`.
    - Train: `R² ≈ 0.985`, `RMSE ≈ 0.94`, `MAPE ≈ 3.39%`.
  - **all / all_no_leak** (NDVI + `crop_id`/`prev_crop_id` + геометрия):
    - CV: `R² ≈ 0.218`, `RMSE ≈ 5.48`, `MAE ≈ 3.99`, `MAPE ≈ 17.8%`.
    - Train: `R² ≈ 0.993`, `RMSE ≈ 0.66`, `MAPE ≈ 2.38%`.
    - **Это сейчас лучшая комбинация по CV**.

- **Датасет `ops_ndvi_2021_2025`** (`ml_dataset_ops_ndvi_2021_2025.csv`) — **маленький, сильно переобучается**
  - Любая модель (no_ndvi, ndvi_only, all, all_no_leak):
    - Train: `R² ≈ 0.999–1.0`, `RMSE ≪ 1 т/га`, `MAPE < 1%` → модель «зазубрила» 136 точек.
    - CV: `R² сильно отрицательный` (от `-0.3` до `-3.1` в среднем, std огромный) → обобщения почти нет.
  - Практический вывод: **использовать этот датасет для анализа важности признаков**, а не как финальную модель качества.

### 4. Список файлов проекта (кратко, «за что отвечает»)

- **Корень проекта**
  - `build_ml_dataset.py` — главный скрипт подготовки данных:
    - строит `ml_dataset_with_ndvi.csv`;
    - формирует `targets_extended.csv`;
    - строит `ml_dataset_ndvi_only_extended.csv` и `ml_dataset_ops_ndvi_2021_2025.csv`;
    - создаёт все `field_*`, `ops_*`, `ndvi_*`, `crop_id`, `prev_crop_id` и делает импутацию пропусков.
  - `train_baseline_model.py` — тренинг и сравнение моделей:
    - грузит все три датасета;
    - для каждого гоняет 4–5 вариантов моделей (`no_ndvi`, `ndvi_only`, `all`, `*_no_leak`);
    - считает CV‑метрики (GroupKFold по `year`);
    - сохраняет модели в `models/` и таблицу `models/models_comparison.csv`;
    - рисует `feature_importance_*.png` и `predictions_plot_*.png`.
  - `fetch_cropwise_data.py` — базовая выкачка из Operations API:
    - `/fields`, `/crops`, `/operations`, `/yield_maps`, `/productivity_estimates`, `/historical_values?type=ndvi` и т.п.;
    - умеет писать CSV в `data_raw/` с флагами, какие сущности забирать.
  - `fetch_ndvi_timeseries.py` — отдельный скрипт для выкачки NDVI time‑series (через `/api/v3a/historical_values?type=ndvi`).
  - `fetch_history_items_extended.py` — выкачка всех `history_items` (с пагинацией) и построение:
    - `data_raw/history_items_full.csv`;
    - `data_raw/additional_yields_from_history.csv` (доп. таргет).
  - `fetch_fertilizers_npk.py` — выкачка `/fertilizers` и разбор поля `elements` в `fertilizers_npk.csv`.
  - `fetch_seeds_catalog.py` — выкачка `/seeds/{id}` для уникальных `seed_id` из операций, создание `seeds.csv`.
  - `README_ML.md` — how‑to по запуску ML‑пайплайна.
  - `REPORT_FOR_PERPLEXITY.md` — развернутый текстовый отчёт/ТЗ (для внешних).
  - `requirements.txt` — список зависимостей (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `requests`).

- **`data_raw/`**
  - `fields.csv` — геометрия и метаданные полей.
  - `crops.csv` — справочник культур.
  - `operations.csv` — все агрооперации по полям/сезонам.
  - `productivity_estimates.csv` — базовый таргет по полям/годам.
  - `history_items_full.csv` — полная история полей (культуры, урожайности, даты).
  - `additional_yields_from_history.csv` — новые `field_id×year` с урожайностью, которых нет в `productivity_estimates`.
  - `ndvi_timeseries.csv` — все наблюдения NDVI по полям/датам.
  - `fertilizers_npk.csv` — справочник удобрений с NPK‑составом.
  - `seeds.csv` — справочник семян (для маппинга `seed_id → crop_id`).
  - `yield_maps.csv` — метаданные карт урожайности.
  - `weather_stations.csv` — виртуальные метеостанции.
  - `field_to_station_mapping.csv` — соответствие поле→ближайшая станция.

- **`data_processed/`**
  - `ml_dataset_with_ndvi.csv` — базовый ML‑датасет (440 строк).
  - `targets_extended.csv` — расширенный таргет (все известные `field_id×year` с урожайностью).
  - `ml_dataset_ndvi_only_extended.csv` — NDVI‑only + геометрия + `crop_id`/`prev_crop_id` (441 строка).
  - `ml_dataset_ops_ndvi_2021_2025.csv` — только годы с операциями, все `ops_*` + NDVI (136 строк, 2021–2025).
  - (возможные дополнительные файлы `*_v2`, `seed_id_to_crop_id.csv` — экспериментальные артефакты).

- **`models/`**
  - `models_comparison.csv` — сводная таблица всех моделей (см. блок 3).
  - `model_<dataset>_<model_type>.pkl` — сериализованные `GradientBoostingRegressor` для каждой комбинации.
  - `feature_importance_<dataset>_<model_type>.png` — bar‑plot важности признаков.
  - `predictions_plot_<dataset>_<model_type>.png` — scatter `y_true vs y_pred` + residuals.
  - Старые файлы без префикса датасета (`model_all.pkl`, `feature_importance_all.png` и т.д.) — legacy‑артефакты от старой версии `train_baseline_model.py`.

---

Эта картинка отражает текущее состояние: какие данные у нас есть, какие датасеты собраны, какие модели уже обучены и как они себя ведут по CV. Дальше можно решать, куда развивать фичи (NDVI‑интеграл/фазы, предшественники, погодные индексы, soil), и какие сценарии/ключевые модели реально тянуть в прод (например, `ndvi_only_extended / all_no_leak` как базовую линию). 

