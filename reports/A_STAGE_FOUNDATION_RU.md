# A-stage foundation: target policy + feature contract

Цель этапа A: убрать методический шум перед добавлением новых данных
(Sentinel-2 red-edge/EVI/GCVI и менеджмент).

## A1. Low-yield target audit

Файл: `reports/A1_LOW_YIELD_AUDIT_RU.md`.

Основная находка:

- Всего строк v24: **1 903**.
- Нормальные строки `target >= 1 т/га`: **1 684** (88.5%).
- Low-yield строки `target < 1 т/га`: **219** (11.5%).
- Явных real crop failure по NDVI: **0**.
- Подозрительных target rows с высоким NDVI/Cropwise evidence: **167**.

Это значит: low-yield строки в основном не похожи на реальные погибшие посевы.
На них NDVI часто нормальный (`median ndvi_max_asof = 0.647`), то есть поле
выглядело как нормальный crop canopy, но target оказался <1 т/га.

### Решение

Main yield regression и сравнение с Cropwise считаем на:

```text
target_yield_t_ha >= 1.0
```

Low-yield строки не удаляем из проекта, но ведём отдельно:

- target-quality audit;
- crop failure / anomaly detection;
- примеры для методологии диссертации.

Так мы не прячем проблему, но и не смешиваем разные задачи в один MAPE.

## A2. Feature pruning / contract

Создано:

- `scripts/asof/feature_sets_v24.py`
- `scripts/asof/evaluate_v24_pruned_features.py`
- `reports/A2_V24_PRUNED_FEATURES_RU.md`

Проверили компактный агро-набор признаков:

- NDVI core;
- root-zone soil moisture;
- GDD / heat stress;
- soil P/K/OM/pH;
- sowing / rotation;
- без мёртвых `region_id`, `district_id`, `till_type`, `lat`, `long`.

### Результат

Compact/pruned набор **не универсально лучше**:

- `wheat_combined`, 1 Sep: MAPE улучшился с **33.3%** до **30.6%**.
- `all_crops`, 1 Sep: MAPE улучшился с **32.8%** до **31.7%**.
- `sunflower`: pruned ухудшил результат, полный v24 feature pool лучше.

### Решение

Не заменяем полный v24 набор автоматически.

Фиксируем два режима:

1. **Full v24 feature pool** — основной для sunflower и ранних дат.
2. **Pruned/core feature set** — кандидат для wheat / поздней даты / регуляризации.

В следующих версиях выбираем режим per crop × as-of по walk-forward validation.

## Что это меняет в плане

Этап A завершён.

Теперь фундамент чистый:

- main метрика не ломается low-yield target artifacts;
- понятно, какие признаки мёртвые;
- есть компактный feature contract;
- но финальный выбор full/pruned делается валидацией.

Следующий этап по roadmap:

```text
B1. Sentinel-2 red-edge/EVI/GCVI по геометрии 640 полей
```

Почему именно B1:

- Cropwise API не отдаёт NDRE/EVI/GCVI;
- литература показывает, что red-edge/EVI/GCVI помогают там, где NDVI насыщается;
- это самый вероятный рычаг для пшеницы и late-season 1 Sep, где Cropwise пока сильнее.
