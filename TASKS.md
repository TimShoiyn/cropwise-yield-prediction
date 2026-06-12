# TASKS — Cropwise Yield Prediction Cleanup

> Чек-лист по PLAN.md. Каждая задача либо `[ ]` либо `[x]` либо `[~]` (in progress).

---

## Sprint 1: Очистка таргета и данных

### 1.1 Clean targets
- [ ] `scripts/clean/build_clean_targets.py`
  - [ ] Загрузить `productivity_estimates.csv`, `history_items_full.csv`, `fields.csv`, `crops.csv`
  - [ ] Реализовать physical target: `harvested_weight / tillable_area` (т/га, ~129 строк)
  - [ ] Реализовать per-crop unit normalization (потолки из PLAN.md §2.1)
  - [ ] Outlier removal: P5/P99.5 per crop, drop 0
  - [ ] Save `data_processed/targets_cleaned_t_ha.csv`
  - [ ] Save audit log `data_processed/targets_cleaned_audit.csv` (что/как/почему конвертировалось)
- [ ] `reports/target_cleaning_audit.md` — сводка по культурам с до/после

### 1.2 Clean dataset
- [ ] `scripts/clean/build_clean_dataset.py`
  - [ ] База: `ml_dataset_full_extended_cropwindow_v1_phases.csv`
  - [ ] Replace target c `targets_cleaned_t_ha.csv`
  - [ ] Drop фичи: `field_lat`, `field_long`, `field_group_id`, `field_soil_CEC`, `field_soil_Ca_saturation`, `field_soil_N_NO3`, harvest-leakage (`ops_count_harvesting`, `ops_days_seeding_to_harvest`, `ops_season_end`, `ops_season_duration_days`, `ops_yield_t_ha`)
  - [ ] Save `data_processed/ml_dataset_clean_v2.csv`

### 1.3 EDA report
- [ ] `scripts/analysis/audit_clean_dataset.py`
  - [ ] Target distribution per crop (boxplot PNG)
  - [ ] NaN heatmap
  - [ ] Correlation top-30 features × target (per crop)
  - [ ] Year × n_rows table
- [ ] `reports/clean_dataset_audit.md`

---

## Sprint 2: Честный baseline

### 2.1 CatBoost on clean data
- [ ] `scripts/train/train_clean_catboost.py`
  - [ ] Walk-Forward CV (test_year от 2017 до 2025, train ≥ 5 предыдущих лет)
  - [ ] CatBoost с `cat_features=['crop_id', 'prev_crop_id', 'field_id']`
  - [ ] OOF predictions, feature importance, SHAP top-20
  - [ ] Save `models_v2/catboost_clean_allcrops/`

### 2.2 Per-crop models
- [ ] sunflower (n≈280)
- [ ] wheat_combined (n≈100)

### 2.3 Comparison
- [ ] `reports/model_comparison_v1_vs_v2.md`

---

## Sprint 3: As-of forecast (главный научный вклад)

### 3.1 As-of dataset
- [ ] `scripts/asof/build_asof_dataset.py`
  - [ ] Для D ∈ {2025-07-01, 2025-08-01, 2025-09-01} (и аналогично для каждого года) обрезать NDVI и weather на D
  - [ ] Save `data_processed/ml_dataset_clean_v2_asof_07.csv`, `_asof_08.csv`, `_asof_09.csv`

### 3.2 Train + benchmark
- [ ] `scripts/asof/train_asof.py` — модель для каждой D
- [ ] `scripts/asof/extract_cropwise_asof.py` — достать прогнозы Cropwise на даты D из `productivity_estimate_histories.csv`
- [ ] `scripts/asof/compare_asof.py` — сравнить ML vs Cropwise

### 3.3 Plots
- [ ] График: дата прогноза × MAPE, линии ML и Cropwise (sunflower)
- [ ] График: дата прогноза × MAPE, линии ML и Cropwise (wheat)
- [ ] `reports/asof_forecast_results.md`

---

## Sprint 3.5: Агрономические фичи v2 (если время есть)

- [ ] `scripts/features/agro_features.py`
  - [ ] GDD от даты сева (Tbase: пшеница=5, подсолнечник=6, ячмень=5, кукуруза=10)
  - [ ] DSI/HSI в критические фазы (пшеница: налив зерна; подсолнечник: цветение)
  - [ ] `crop_pair` = (prev_crop, current_crop)
  - [ ] `years_since_sunflower` per field

---

## Sprint 4: Финал

- [ ] `reports/FINAL_RESULTS.md` со всеми сравнениями
- [ ] Главные графики PNG для статьи (3-4 шт)
- [ ] Update `текст для статьи метод и получение результатов статья скр тхт.txt` или новый Markdown с честными цифрами
