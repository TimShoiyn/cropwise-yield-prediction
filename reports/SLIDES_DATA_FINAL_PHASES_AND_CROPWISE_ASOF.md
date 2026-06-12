## TL;DR (что вставлять в слайд)

### Моя модель (allcrops, честная CV по годам, OOF)
- **Baseline (сезонные агрегаты, без фаз)**: \(R^2 = 0.309\), RMSE \(= 5.39\) т/га, MAPE \(= 17.9\%\).
- **Phases (фазные GDD‑фичи + NDVI по фазам)**: \(R^2 = 0.372\), RMSE \(= 5.16\) т/га, MAPE \(= 17.2\%\).
- **Вывод**: фазные признаки дали **+0.063 к \(R^2\)** и снизили RMSE примерно на **0.23 т/га**.

Источник: `models/models_phases_summary_short.csv` (или `models/models_phases_comparison.csv`).

### Cropwise (честные прогнозы во времени из API, estimate_history)
По `models/cropwise_asof_metrics.csv`:
- **as-of 01.07**: \(R^2 = 0.324\), RMSE \(= 6.32\) т/га, MAPE \(= 20.8\%\)
- **as-of 01.08**: \(R^2 = 0.734\), RMSE \(= 3.96\) т/га, MAPE \(= 12.8\%\)
- **as-of 01.09**: \(R^2 = 0.904\), RMSE \(= 2.38\) т/га, MAPE \(= 6.88\%\)
- **final**: \(R^2 = 1.0\), RMSE \(= 0\), MAPE \(= 0\%\) — **не честный forecast**, т.к. финальное значение фактически совпадает с фактом.

### Моя модель vs Cropwise as-of (кто лучше и когда)
По `models/cropwise_asof_vs_model_metrics.csv` (сравнение по RMSE, ниже лучше):
- **01.07**: моя модель лучше — **5.60 vs 6.32**, выигрыш \(\approx 0.71\) т/га
- **01.08**: Cropwise лучше — **3.96 vs 5.60**, выигрыш \(\approx 1.64\) т/га
- **01.09**: Cropwise существенно лучше — **2.38 vs 5.60**, выигрыш \(\approx 3.22\) т/га

---

## Что мы сделали (краткая история для научрука/НИР)

1) Построили ML‑модель прогноза урожайности на уровне **field×year** на данных NDVI+погоды+почвы.  
Валидация: **GroupKFold по `year`**, оценка по **OOF** (честно, без утечек).

2) Добавили фазные признаки по GDD (3 фазы) и проверили эффект:
- На **allcrops** качество выросло.
- На **sunflower** и **wheat** качество в текущей версии ухудшилось (неоднородность эффекта по культурам).

3) Обнаружили, что “итоговый” CSV `productivity_data.csv` не годится как независимый прогноз (там прогноз совпадает с фактом на совпавших строках).

4) Вытащили из API реальные временные ряды прогнозов Cropwise:
- `productivity_estimate_histories.csv` содержит `estimate_history` (date → predicted yield).
Построили честные as-of прогнозы на 01.07 / 01.08 / 01.09 и сравнили с фактом и с нашей моделью.

---

## Предлагаемый формат 2–3 слайдов (готовый “скелет”)

### Слайд 1 — “Моя модель: baseline vs phases (честная CV по годам)”
**Цель:** улучшить прогноз урожайности на уровне поле–год, используя NDVI и погодные данные.

**Результат (allcrops, OOF, GroupKFold by year):**
- Baseline: \(R^2 = 0.309\), RMSE \(= 5.39\), MAPE \(= 17.9\%\)
- Phases: \(R^2 = 0.372\), RMSE \(= 5.16\), MAPE \(= 17.2\%\)

**Вывод:** фазные признаки улучшают общую модель (**+0.063 к \(R^2\)**).

**Вставить визуализации:**
- `models/plots/phases_cv_r2_barplot_3subsets.png`
- `models/plots/phases_feature_importance_allcrops_phases_top20.png`
- (опционально) `models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png`

### Слайд 2 — “Cropwise: честный прогноз во времени (as-of из API)”
**Данные:** `estimate_history` из `/api/v3/productivity_estimate_histories`.

**Метрики Cropwise vs факт (field×year):**
- 01.07: \(R^2 = 0.324\), RMSE \(= 6.32\), MAPE \(= 20.8\%\)
- 01.08: \(R^2 = 0.734\), RMSE \(= 3.96\), MAPE \(= 12.8\%\)
- 01.09: \(R^2 = 0.904\), RMSE \(= 2.38\), MAPE \(= 6.88\%\)

**Комментарий:** точность Cropwise резко растёт к августу–сентябрю (регулярные обновления внутри сезона).

**Где лежат цифры:**
- `models/cropwise_asof_metrics.csv`

### Слайд 3 — “Моя модель vs Cropwise as-of (кто лучше и когда)”
**Сравнение по RMSE:**
- 01.07: модель лучше на \(\approx 0.71\) т/га
- 01.08: Cropwise лучше на \(\approx 1.64\) т/га
- 01.09: Cropwise лучше на \(\approx 3.22\) т/га

**Интерпретация:**
- Моя модель даёт “ранний” сезонный прогноз по агрегатам NDVI/погоды и выигрывает в начале сезона.
- Cropwise выигрывает позже (к августу/сентябрю), т.к. использует обновления данных внутри сезона (estimate_history).

**Где лежат цифры:**
- `models/cropwise_asof_vs_model_metrics.csv`

---

## Пути ко всем результатам, которые сделали “только что”

### Фазные фичи и сравнение baseline vs phases
- Датасет с фазами: `data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv`
- Метрики (полная таблица): `models/models_phases_comparison.csv`
- Метрики (коротко): `models/models_phases_summary_short.csv`
- Интерпретация (короткий текст): `models/phases_interpretation_short.txt`
- OOF (allcrops baseline/phases):  
  - `models/oof_phases_baseline_allcrops.csv`  
  - `models/oof_phases_phases_allcrops.csv`
- Графики:
  - `models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png`
  - `models/plots/phases_cv_r2_barplot_3subsets.png`
  - `models/plots/phases_feature_importance_allcrops_phases_top20.png`

### Model vs Cropwise (по “итоговому” productivity_data.csv)
> Важно: “прогноз” в `productivity_data.csv` на совпавших строках совпадает с фактом → \(R^2=1\), это не честный прогноз.
- Join: `models/models_vs_cropwise_joined_field_year.csv`
- Метрики: `models/models_vs_cropwise_metrics.csv`

### Cropwise честные as-of прогнозы из API (estimate_history)
- Сырые данные из API (уже выгружено): `data_raw/productivity_estimate_histories.csv`
- Таблица field×year с as-of прогнозами + факт: `models/cropwise_asof_joined.csv`
- Метрики as-of: `models/cropwise_asof_metrics.csv`

### Cropwise as-of vs моя модель (OOF)
- Join: `models/cropwise_asof_vs_model_joined.csv`
- Метрики: `models/cropwise_asof_vs_model_metrics.csv`

---

## Команды для воспроизводимости (если спросят “как получено”)

- Фазы (датасет + обучение): `python train_phases_experiment.py`
- Артефакты фаз (графики/summary): `python make_phases_artifacts.py`
- Проверка productivity_data vs ML (и метрики): `python compare_model_vs_cropwise.py`
- Cropwise as-of из estimate_history: `python cropwise_asof_eval.py`
- Cropwise as-of vs модель: `python cropwise_asof_vs_model.py`

