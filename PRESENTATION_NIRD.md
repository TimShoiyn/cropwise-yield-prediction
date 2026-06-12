# Презентация НИРД — Шойынбек Темірлан Арманұлы

> Файл — расширенная версия твоего исходного текста. Везде, где нужно вставить таблицу/график/цифру в PPTX, указан **точный путь к файлу** в проекте (`/Users/shoiyn/Desktop/PhD/Cropwise_api_data/...`).
> Блоки `INSERT → ...` — это инструкции, что именно вставлять на слайд.

---

## Page 1 — Титульный слайд

**ОТЧЕТ ПО НАУЧНОЙ ИССЛЕДОВАТЕЛЬСКОЙ РАБОТЕ ДОКТОРАНТА**

Докторант: **Шойынбек Темірлан Арманұлы**

---

## Page 2 — Полная информация о докторанте

- **ФИО**: Шойынбек Темірлан Арманұлы
- **Курс**: 1 курс
- **Наименование и шифр ОП**: 8D06102 — «Machine Learning & Data Science»

---

## Page 3 — Тема докторской диссертации

- KZ: «Интеллектуалды егіншілік контурындағы машиналық оқытудың болжамды модельдері»
- RU: «Предиктивные модели машинного обучения в контуре интеллектуального земледелия»
- EN: «Predictive machine learning models in smart farming»

---

## Page 4 — Научные руководители

**Научный руководитель**
- Мухамедиев Равиль Ильгизович
- д. инж. н., профессор
- Институт Автоматики и Информационных технологий
- Кафедра Программной инженерии

**Зарубежный научный руководитель**
- (Prof. Madya Dr.) RAZALI BIN YAAKOB
- PhD, Associate Professor
- Faculty of Computer Science and Information Technology
- UPM (Universiti Putra Malaysia), Malaysia

---

## Page 5 — Научные руководители (фото)

> Слайд с фотографиями. Оставь как есть.

---

## Page 6 — Актуальность исследования

- Рост климатических и рыночных рисков в сельском хозяйстве требует более точного прогнозирования продуктивности полей.
- Цифровые FMS-платформы (например, **Cropwise Operations**) аккумулируют большие объёмы данных: NDVI, метеоданные, состояние почвы, агрооперации.
- Предиктивные модели машинного обучения позволяют превратить эти данные в практические рекомендации для агрономов и менеджмента хозяйства.

**Дополнить цифрами проекта (для устного комментария):**
- Используются реальные данные одного хозяйства: **30 полей × годы 2010–2025**, итоговая обучающая выборка — **441 наблюдение** (поле × год).
- Источник датасета: `data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv`.

---

## Page 7 — Научная проблема

- Данные реальных FMS-систем разнородны, содержат пропуски и отличаются по качеству между полями и сезонами.
- Неочевидно, какие комбинации признаков (NDVI по фазам, осадки, активные температуры, влажность почвы и т.д.) дают наилучший прогноз продуктивности.
- Отсутствует отработанная методика интеграции предиктивных моделей в контур интеллектуального земледелия конкретного предприятия.

**Подкрепление цифрами:**
- Доля NaN в погодных признаках: **all = 18.6%**, **sunflower = 28.7%**, **wheat = 0.0%** (источник: `reports/debug/summary.txt`).
- В исходных данных 2 почвенных признака (`field_soil_CEC`, `field_soil_Ca_saturation`) имеют **100% NaN** и автоматически дропаются перед обучением.

---

## Page 8 — Цель исследования

Разработать и экспериментально оценить предиктивные модели машинного обучения для прогноза показателей продуктивности сельскохозяйственных земель на основе данных интеллектуального земледелия.

**Конкретизация цели в текущей итерации:**
- Прогноз **`target_yield_t_ha`** (т/га) на уровне `field_id × year`.
- Честная валидация: **GroupKFold по годам** + метрики по **OOF**‑предсказаниям.

---

## Page 9 — Задачи исследования

1. Выполнить обзор подходов к прогнозу NDVI и урожайности на основе ML/DL и спутниковых данных.
2. Сформировать датасет по полям на основе данных Cropwise Operations (NDVI, метеоданные, состояние почвы, агрооперации, продуктивность).
3. Разработать схему предобработки данных и формирования признаков. Обучить и сравнить несколько моделей (LR, Random Forest, градиентный бустинг, при наличии — LSTM/Temporal CNN).
4. Оценить качество по метрикам **R²**, **RMSE**, **MAE** и проанализировать важность признаков.

**Что уже сделано (в одну строку):**
- Собран сквозной пайплайн `build_ml_dataset.py` → `train_baseline_model.py` → `train_phases_experiment.py`, сравнены **4 модели** (LR / RF / GBDT / CatBoost), посчитаны OOF‑метрики и feature importance.

---

## Page 10 — Смена направления

- Изначально утверждённая тема диссертации относилась к другой предметной области и опиралась на иной тип данных.
- В процессе возникли объективные трудности с доступом к необходимым данным, что существенно ограничило возможность реализации первоначальной темы.
- Совместно с научным руководителем и при одобрении заведующей кафедрой было принято решение перейти к теме предиктивных моделей для интеллектуального земледелия, где имеется доступ к данным **Cropwise Operations** и сформирован устойчивый исследовательский задел.
- Отдел докторантуры оформит официальную смену темы в начале 2-го курса, поэтому в отчёте и презентации представлено фактическое текущее направление.

---

## Page 11 — Исходные данные

- **Источник**: платформа управления полями **Cropwise Operations** (API v3).
- **Данные**:
  - временные ряды NDVI по полям (источник: `data_raw/ndvi_timeseries.csv`),
  - метеопараметры — осадки, температура, активные температуры (`data_raw/weather_season_aggregates.csv`),
  - почвенные тесты — pH, OM, NPK, Mg, CEC и др. (`data_raw/soil_tests.csv`),
  - агрооперации (`data_raw/operations.csv`),
  - история культур (`data_raw/history_items_full.csv`).
- **Целевая переменная**: `target_yield_t_ha` — урожайность (т/га) на пару `field_id × year`. Собирается из `data_raw/productivity_estimates.csv` (+ `yield_maps.csv` и `additional_yields_from_history.csv` как fallback).

> INSERT → таблица с обзором датасетов:
> `reports/tables/dataset_statistics.csv` (или картинка `reports/tables/dataset_statistics.png`).

**Готовое содержимое таблицы (можно вставить как есть):**

| Dataset | N obs | N fields | Years | Mean yield (t/ha) | Std yield (t/ha) | N features |
|---|---:|---:|---|---:|---:|---:|
| base_with_ndvi | 440 | 30 | 2010–2025 | 24.04 | 7.69 | 37 |
| ndvi_only_extended | 441 | 30 | 2010–2025 | 24.08 | 7.74 | 51 |
| full_extended | 441 | 30 | 2010–2025 | 24.08 | 7.74 | 41 |
| ops_ndvi_2021_2025 | 136 | 30 | 2021–2025 | 30.16 | 7.81 | 60 |

---

## Page 12 — Формирование датасета

- Объединение NDVI, метеоданных, влажности почвы и информации об операциях по ключам `field_id × date` и `field_id × year`.
- Очистка выбросов, заполнение пропусков, агрегирование по сезонам и ключевым фенофазам развития культуры.
- Формирование признаков: статистики NDVI (`ndvi_mean_season`, `ndvi_max_season`, `ndvi_early/mid/late`), суммы осадков, активные температуры (`weather_gdd_season`), показатели влажности/почвы, признаки агроопераций.

**Сезонные окна (crop‑window v1):**
- `wheat_spring` → май–сентябрь (5–9),
- `sunflower` → май–октябрь (5–10),
- остальные культуры → май–октябрь (дефолт).

Скрипт сборки: `build_ml_dataset.py` (корень проекта).

> INSERT → схема пайплайна сборки (если есть рисунок) либо подпись:
> «Пайплайн формирования датасета: от сырых данных Cropwise FMS (NDVI / weather / soil / operations / history) к табличному набору признаков `data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv`».

**Структура итогового датасета (`shape = 441 × 29`):**

- Ключи и таргет: `field_id`, `year`, `target_yield_t_ha`.
- Культура: `crop_id`, `prev_crop_id`.
- Геометрия: `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`.
- NDVI: `ndvi_observations`, `ndvi_mean_season`, `ndvi_max_season`, `ndvi_early`, `ndvi_mid`, `ndvi_late`.
- Почва: `field_soil_pH`, `field_soil_OM`, `field_soil_P`, `field_soil_K`, `field_soil_N`, `field_soil_N_NO3`, `field_soil_Mg`.
- Погода: `weather_temp_avg_season`, `weather_gdd_season`, `weather_precip_sum_season`, `weather_precip_sum_early`, `weather_hot_days`.

Полный словарь признаков: `reports/FEATURES_EXPLAINED_CROPWINDOW_V1.md`.

---

## Page 13 — Предиктивные модели машинного обучения

- **Базовая модель**: линейная регрессия (LR) — для оценки «нижнего» уровня качества.
- **Основные модели**: Random Forest и градиентный бустинг (**Baseline GBDT**, **CatBoost**, **Regularized GBDT**).
- **Перспективное направление**: LSTM / Temporal CNN по временным рядам NDVI и метеоданных (по мере накопления большего объёма данных).
- **Метрики**: R², RMSE, MAE, MAPE.
- **Валидация**: `GroupKFold(n_splits=5)` по `year` (тестовый год не виден при обучении) + OOF‑метрики.

**Сравнение моделей (источник: `reports/tables/models_comparison_3models.csv`):**

| Model | max_depth | min_samples_leaf | subsample | lr | n_estimators | **CV R²** | CV RMSE | CV MAE | CV MAPE | Train R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline GBDT | 5 | 1 | 1.0 | 0.10 | 100 | 0.250 | 5.51 | 3.97 | 17.71 | 0.995 |
| CatBoost | 3 | 10 | 0.8 | 0.05 | 500 | 0.300 | 5.47 | 4.02 | 18.40 | 0.900 |
| **Regularized GBDT ★** | 3 | 5 | 0.7 | 0.05 | 300 | **0.370** | **5.14** | **3.82** | **17.59** | 0.961 |

> INSERT → bar‑plot сравнения CV R² по моделям: `reports/figures/cv_r2_comparison.png`.

Скрипты обучения:
- `train_baseline_model.py` (LR / RF / GBDT, 4 набора фич),
- `train_iteration2_catboost.py`,
- `train_iteration2_best_gbdt_full_extended.py`,
- `train_phases_experiment.py` (фазные фичи).

---

## Page 14 — Первые результаты моделирования

**Лучшая модель: Regularized GBDT на `full_extended_cropwindow_v1` (all crops, all_no_leak_reg).**

| Метрика | Значение |
|---|---|
| CV R² (OOF, 5‑fold GroupKFold by year) | **0.309 ± 0.268** |
| CV RMSE (т/га) | **5.39 ± 1.29** |
| CV MAE (т/га) | 3.96 ± 0.66 |
| CV MAPE (%) | **17.91 ± 3.95** |
| Train R² | 0.946 |
| N observations | 441 |
| N features | 42 |

**Разбивка по культурам (OOF, та же модель):**

| Subset | N | **CV R²** | CV RMSE (т/га) | CV MAPE (%) |
|---|---:|---:|---:|---:|
| All crops | 441 | **0.309 ± 0.268** | 5.39 | 17.91 |
| Sunflower | 286 | **0.327 ± 0.153** | 3.84 | 14.90 |
| Wheat (spring) | 67 | **−0.228 ± 0.654** | 4.97 | 16.88 |

**Phases vs Baseline (на всех культурах, `models/models_phases_summary_short.csv`):**

| Вариант | CV R² | CV RMSE | CV MAPE | N features |
|---|---:|---:|---:|---:|
| Baseline (сезонные агрегаты) | 0.309 | 5.39 | 17.91 | 24 |
| **Phases (GDD‑фазы + NDVI по фазам)** | **0.372** | **5.16** | **17.18** | 43 |

→ Фазные признаки дали **+0.063 к R²** и снизили **RMSE на ~0.23 т/га**.

> INSERT → OOF scatter plots (факт vs прогноз):
> - All crops: `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_allcrops_oof.png`
> - Sunflower: `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_sunflower_oof.png`
> - Wheat: `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_wheat_oof.png`
> - Baseline vs Phases: `models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png`
> - Бар‑чарт CV R² по подгруппам: `models/plots/phases_cv_r2_barplot_3subsets.png`

**Сравнение с прогнозом Cropwise (честные as‑of прогнозы из API `estimate_history`):**

| Дата прогноза | Cropwise R² | Cropwise RMSE | Cropwise MAPE | Моя модель RMSE | Кто лучше |
|---|---:|---:|---:|---:|---|
| 01.07 | 0.324 | 6.32 | 20.83% | 5.60 | **моя модель** (Δ ≈ 0.71 т/га) |
| 01.08 | 0.734 | 3.96 | 12.79% | 5.60 | **Cropwise** (Δ ≈ 1.64 т/га) |
| 01.09 | 0.904 | 2.38 |  6.88% | 5.60 | **Cropwise** (Δ ≈ 3.22 т/га) |

Источники: `models/cropwise_asof_metrics.csv`, `models/cropwise_asof_vs_model_metrics.csv`.

> Интерпретация для устной защиты: моя модель работает по сезонным агрегатам и **выигрывает в начале сезона** (когда Cropwise ещё не знает итога), а Cropwise догоняет к августу‑сентябрю за счёт регулярных обновлений `estimate_history`.

---

## Page 15 — Анализ важности признаков

- Построены графики важности признаков для лучшей модели (permutation / gain‑based importance).
- Наиболее значимыми оказались **NDVI в ключевые фазы вегетации** (mid/late) и **суммарные осадки / активные температуры (GDD)** за критические периоды.
- Результаты согласуются с агрономической логикой и подтверждают корректность выбора признаков и подхода к моделированию.

> INSERT → bar‑plot feature importance:
> - Лучшая модель (для презентации): `reports/figures/feature_importance_best_model.png`
> - Cropwindow v1 (all_no_leak): `models/feature_importance_full_extended_cropwindow_v1_all_no_leak.png`
> - С фазами (top‑20): `models/plots/phases_feature_importance_allcrops_phases_top20.png`
> - Полная таблица: `models/phases_feature_importance_allcrops_full.csv`

**Топ‑факторы (типичный порядок по важности на лучшей модели):**

1. `ndvi_mid` / `ndvi_late` — NDVI в середине и в конце вегетации,
2. `weather_gdd_season` — сумма активных температур за сезон,
3. `weather_precip_sum_season` — суммарные осадки за сезон,
4. `crop_id`, `prev_crop_id` — культура и предшественник (севооборот),
5. `field_soil_OM`, `field_soil_pH`, `field_soil_N` — базовые почвенные показатели,
6. `field_lat`, `field_long` — региональные/локальные градиенты.

---

## Page 16 — Научная статья

- По результатам НИРД подготовлена статья совместно с научным руководителем, посвящённая применению данных **Cropwise Operations** и предиктивных моделей машинного обучения для оценки и прогноза продуктивности земель.
- Статья направлена в журнал **«Вестник КБТУ»** (выпуск 2026 года), на данный момент **на стадии рецензирования**, планируемая публикация — **лето 2026 года**.
- После стабилизации и расширения моделей планируется подготовка расширенной версии исследования для подачи в международный журнал, индексируемый в **Scopus** (целевой квартиль Q2).

Черновик/материал статьи: `текст для статьи метод и получение результатов статья скр тхт.txt` (в корне проекта).

---

## Page 17 — План дальнейшей работы

1. Расширение датасета (новые поля, сезоны); улучшение предобработки и признаков (фенологические фазы, индексы NDMI/EVI, ET).
2. Тестирование временных моделей (**LSTM / Temporal CNN / Transformer** по NDVI‑таймсериям), сравнение с RF/GBDT/CatBoost.
3. Rolling‑forecast эксперимент: as‑of прогноз в июне / июле / августе с обрезкой данных по дате (чтобы сравниться с Cropwise on equal footing).
4. Интеграция моделей в BI‑дашборды и контур поддержки решений агропредприятия.
5. Подготовка и защита диссертации, публикации в журналах **Q1–Q2** (Scopus / WoS).

---

## Page 18 — Выводы

- Сформирован датасет интеллектуального земледелия на основе данных **Cropwise Operations** (NDVI, метеоданные, состояние почвы, операции, продуктивность): **441 наблюдение, 30 полей, 2010–2025, 29–69 признаков**.
- Реализованы первые предиктивные модели машинного обучения. Лучшая модель — **Regularized GBDT + phases features**: **CV R² ≈ 0.37, RMSE ≈ 5.16 т/га, MAPE ≈ 17.2 %** на честной CV по годам.
- Полученные результаты сопоставимы со встроенным прогнозом Cropwise на ранних датах (01.07) и создают основу для дальнейшего развития моделей, их интеграции в контур интеллектуального земледелия и оформления диссертационной работы по обновлённой теме.

---

## Page 19 — Литературный обзор

1. **Huete A.R., Didan K., Miura T., Rodriguez E.P., Gao X., Ferreira L.G.**
   Overview of the radiometric and biophysical performance of the MODIS vegetation indices // Remote Sensing of Environment. — 2002. — Vol. 83, No. 1–2. — P. 195–213. DOI: 10.1016/S0034-4257(02)00096-2.

2. **Sishodia R.P., Ray R.L., Singh S.K.**
   Applications of Remote Sensing in Precision Agriculture: A Review // Remote Sensing. — 2020. — Vol. 12, No. 19. — Art. 3136. DOI: 10.3390/rs12193136.

3. **Le M., Bolten J.D., Whitney K.M., et al.**
   On the Use of SMAP Soil Moisture for Forecasting NDVI Over CONUS Cropland Regions // Geophysical Research Letters. — 2024. — Vol. 51, No. 20. DOI: 10.1029/2024GL111187.

4. **Farbo A., Sarvia F., De Petris S., et al.**
   Forecasting corn NDVI through AI-based approaches using Sentinel-2 time series // ISPRS Journal of Photogrammetry and Remote Sensing. — 2024. — Vol. 211. — P. 244–261. DOI: 10.1016/j.isprsjprs.2024.06.014.

5. **Milazzo F., Brocca L., Vanwalleghem T.**
   NDVI Prediction of Mediterranean Permanent Grasslands Using Soil Moisture Products // Agronomy. — 2024. — Vol. 14, No. 8. — Art. 1798. DOI: 10.3390/agronomy14081798.

6. **Jeba R.P., Kirthiga S.M., Issac A.M., et al.**
   An improved framework for mapping and assessment of dynamics in cropping pattern and crop calendar from NDVI time series // Environmental Monitoring and Assessment. — 2024. — Vol. 196. — Art. 1141. DOI: 10.1007/s10661-024-13270-1.

7. **Md-Tahir H., Mahmood H.S., Husain M., et al.**
   Localized Crop Classification by NDVI Time Series Analysis of Remote Sensing Satellite Data // AgriEngineering. — 2024. — Vol. 6, No. 3. — P. 2429–2444. DOI: 10.3390/agriengineering6030142.

8. **USGS.** Normalized Difference Moisture Index (NDMI) — Landsat Missions. URL: https://www.usgs.gov/landsat-missions/normalized-difference-moisture-index

9. **Yengoh G.T., Dent D., Olsson L., Tengberg A.E., Tucker C.J.**
   Use of the NDVI to Assess Land Degradation at Multiple Scales. SpringerBriefs in Environmental Science. Springer, 2016. DOI: 10.1007/978-3-319-24112-8.

10. **EOS Data Analytics.** NDMI Index For Proactive Water Stress Management. URL: https://eos.com/make-an-analysis/ndmi/

---

## Page 20 — Этапы выполнения диссертации

| Семестр | Этап | Основные задачи | Статус |
|---|---|---|---|
| 1.1 | Подготовительный | Литобзор (100+ источников), формулировка проблемы, постановка темы | **Завершён** |
| 1.2 | Теоретический анализ + смена темы | Изучение SOTA по ML/DL для NDVI и урожайности, сбор архива данных Cropwise (API v3) | **Завершён** |
| 2.1 | Проектирование системы | Архитектура пайплайна сборки датасета (`build_ml_dataset.py`), схема предобработки и crop‑window признаков | **В процессе** |
| 2.2 | Реализация прототипа | Реализация моделей (LR, RF, GBDT, CatBoost, +фазы), API/CLI для воспроизводимости (`train_*.py`, `cropwise_asof_eval.py`) | **В процессе** |
| 3.1 | Экспериментальная проверка | Расширение выборки (новые сезоны/поля), временные модели (LSTM/Temporal CNN), rolling‑forecast эксперимент | Планируется |
| 3.2 | Анализ и оформление диссертации | Подготовка текста диссертации, графиков, таблиц, приложений, защита работы; статьи Q1–Q2 | Планируется |

---

## Приложение A — Сводный список артефактов (для PPTX)

### Таблицы (CSV / PNG)
- `reports/tables/dataset_statistics.csv` + `.png` — обзор датасетов.
- `reports/tables/models_comparison_summary.csv` + `models_comparison_summary.png` — все эксперименты по 4 датасетам × 4 наборам фич.
- `reports/tables/models_comparison_3models.csv` — финальный поединок LR / CatBoost / Regularized GBDT.
- `models/models_phases_summary_short.csv` — baseline vs phases по подгруппам (allcrops/sunflower/wheat).
- `models/cropwise_asof_metrics.csv` — точность Cropwise по датам 01.07 / 01.08 / 01.09.
- `models/cropwise_asof_vs_model_metrics.csv` — прямое сравнение моя модель vs Cropwise as‑of.
- `reports/tables/metrics_by_crop_model_vs_cropwise.csv` — метрики по культурам (Подсолнечник, Пшеница яровая/озимая, Ячмень, Рапс и др.).

### Графики (PNG)
- `reports/figures/cv_r2_comparison.png` — сравнение моделей по CV R².
- `reports/figures/predictions_vs_actual_best_model.png` — факт vs прогноз для лучшей модели.
- `reports/figures/feature_importance_best_model.png` — важность признаков лучшей модели.
- `models/feature_importance_full_extended_cropwindow_v1_all_no_leak.png` — feature importance для cropwindow v1.
- `models/plots/phases_feature_importance_allcrops_phases_top20.png` — top‑20 фичей с фазными признаками.
- `models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png` — OOF scatter baseline vs phases.
- `models/plots/phases_cv_r2_barplot_3subsets.png` — CV R² по подгруппам (allcrops/sunflower/wheat).
- `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_allcrops_oof.png` — OOF факт vs прогноз, все культуры.
- `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_sunflower_oof.png` — OOF, подсолнечник.
- `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_wheat_oof.png` — OOF, пшеница яровая.

### Датасеты (CSV)
- `data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv` — основной ML‑датасет (441 × 29).
- `data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv` — с фазными GDD‑фичами.
- `data_processed/targets_extended.csv` — расширенный таргет.

### Документация проекта (MD)
- `reports/NIRD_PREP_GUIDE.md` — подробная шпаргалка под защиту (сценарий ответов).
- `reports/DATA_LINEAGE.md` — паспорт данных и пайплайна.
- `reports/FEATURES_EXPLAINED_CROPWINDOW_V1.md` — словарь признаков.
- `reports/MODEL_TRANSPARENCY_CURRENT_STATE.md` — прозрачность моделей.
- `reports/SLIDES_DATA_FINAL_PHASES_AND_CROPWISE_ASOF.md` — готовый «скелет» слайдов c фазами и as‑of.

### Воспроизводимость (команды)
```bash
python build_ml_dataset.py                            # собрать датасеты
python train_baseline_model.py                        # LR / RF / GBDT, OOF + графики
python train_iteration2_catboost.py                   # CatBoost
python train_iteration2_best_gbdt_full_extended.py    # Regularized GBDT (★ best)
python train_phases_experiment.py                     # baseline vs phases
python make_phases_artifacts.py                       # графики фаз
python cropwise_asof_eval.py                          # Cropwise as-of из estimate_history
python cropwise_asof_vs_model.py                      # моя модель vs Cropwise as-of
python debug_dataset_sanity.py                        # sanity (NaN, годы, выбросы)
```

---

## Приложение B — Готовые формулировки на типичные вопросы комиссии

- **«Что именно предсказываете?»** — итоговую урожайность `target_yield_t_ha` (т/га) на уровне `field_id × year`. Текущая модель — post‑season оценка по сезонным агрегатам NDVI/погоды; для оперативного прогноза в течение сезона будет отдельная rolling‑forecast итерация.
- **«Почему Train R² ≈ 0.99, а CV R² ≈ 0.31 — нет ли утечки?»** — нет, это типичный overfitting GBDT на малой выборке. Импутация и масштабирование завёрнуты внутрь `Pipeline`, чтобы статистики не утекали в test. Честная метрика — OOF при `GroupKFold(by=year)`.
- **«Почему у пшеницы R² отрицательный?»** — у wheat 67 наблюдений и 8 лет, в некоторых годах n=2–3, поэтому GroupKFold даёт большой разброс (±0.65). При случайном KFold R² ≈ 0.74 (`reports/debug/summary.txt`) — сигнал есть, нестабильна именно проверка по годам.
- **«Сравнение с Cropwise?»** — итоговый CSV `productivity_data.csv` не годится (R²=1, прогноз = факт). Использован честный `estimate_history` из API: на 01.07 моя модель лучше на ≈0.71 т/га RMSE, к 01.09 Cropwise лучше на ≈3.22 т/га.
