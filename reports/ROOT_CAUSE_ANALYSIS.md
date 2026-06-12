# Root cause analysis — почему модель не выигрывает у Cropwise

Сгенерировано после deep audit (`scripts/audit/deep_data_audit.py`,
`scripts/audit/deeper_investigation.py`, `scripts/audit/coverage_mismatch.py`).

## TL;DR

Главный bottleneck — **NDVI + Open-Meteo собраны только для 30 из 633 полей.**
Наш v7/v9 датасет использует **201 строку / 30 полей**, тогда как доступны
**2067 фактических target-строк по 537 полям** (×10 больше).

Кроме этого:

1. `productivity_data.csv` (3070 строк, факт+прогноз по ВКО) — не использовался.
2. `field_scout_reports_aggregated.csv` (2636 наблюдений, BBCH + threats) — не использовался.
3. `yield_maps.csv` (104 maps с реального комбайна) — не использовался.
4. `soil_test_samples.csv` (1977 sample-строк с pH, NPK, OM, S и пр.) — использовалась только агрегация.
5. NDVI: `cloud_coverage` и `data_coverage` в файле **полностью NaN** —
   фильтр «≤ 30 % облаков» никогда не срабатывал.
6. На 16 строках 2022 г. crop-лейбл `wheat_spring` стоит на озимой
   пшенице (посев август-сентябрь 2021). Это ошибка в данных Cropwise.
7. Sowing dates у sunflower заполнены на 8.5 %, у oil_seed_raps_spring — на 6.3 %.
   v9 sowing-features почти не работают на этих культурах.
8. 2020 — год, где Cropwise сильно провалился (MAE 1.72 t/ha, MAPE 64 %).
   В 2024 — лучший год Cropwise (MAPE 16 %). Наш шанс — годы с
   аномалиями погоды.

## Cardinality table

| источник                              | n field_id | примечание |
|---|---|---|
| history_items_full.csv                | 633 | 2103 строк с productivity |
| targets_factual_t_ha (kept=True)      | 537 | **2067 строк** доступно как target |
| soil_tests                            | 421 | 680 тестов |
| field_scout_reports_aggregated        | 386 | 2636 отчётов с BBCH |
| **fields.csv**                        | **30**  | геометрия / metadata |
| **ndvi_timeseries.csv**               | **30**  | bottleneck |
| **openmeteo_daily.csv**               | **30**  | bottleneck |
| operations.csv                        | 30  | NPK берётся отсюда |
| yield_maps.csv                        | 26  | реальные карты с комбайна |

Пересечение «target есть, NDVI нет» — **513 полей.**
Пересечение «target есть, NDVI есть» — 30.

## Три варианта дальнейшего движения

### Вариант A — расширить покрытие до всего набора (predprocesssing)

Доскачать NDVI и Open-Meteo для оставшихся 513 полей с target.
Нужны:

- геометрии полей (там где их нет — Cropwise API endpoint `/fields/{id}`,
  судя по структуре `data_raw`),
- NDVI с Open Platform API,
- ERA5-Land с Open-Meteo (~3 ч rate-limited скачивания).

Profit: датасет растёт с 200 → ~2000 строк, можно делать настоящее ML.
Cost: 1-2 дня на скачивание + проверка пересечения с soil/scout.

### Вариант B — выжать максимум из 30 полей

Не трогаем coverage, но добавляем:

- BBCH/threats из `field_scout_reports`,
- per-sample soil chemistry,
- yield_maps как honest validation set,
- multi-year stacking (как в Cropwise, чтобы был «эффект памяти поля»).

Profit: выигрыш на текущих 200 строках.
Cost: 1-2 спринта feature engineering. Но мощного буста ждать не стоит —
30 полей × 7 лет это слишком мало для ML.

### Вариант C — гибрид (рекомендуется)

1. Сначала сделать **B** (BBCH/threats/yield_maps), это быстрый честный буст.
2. Параллельно выяснить, можем ли мы получить геометрии 513 полей.
3. Если да — расширить datasource, если нет — закрепить B и писать статью.

## Что ещё всплыло (полезно для статьи и будущих PR)

- NDVI у нас уже постпроцессированный daily ряд (~365 значений в год на поле,
  cloud_coverage пуст). Поэтому когда мы берём «last30d_mean» — это уже
  гладкий тренд, и большая часть нашего «NDVI feature engineering»
  фактически не несёт нового сигнала. Нужно перейти к dynamics-based:
  скорость снижения NDVI после пика, dose-response к дождю/жаре в течение
  определённого окна.
- Сильный домен подсолнечника: 78 % таргетов на нём — потому что у клиента
  это монокультура поля 195. Это объясняет, почему модель «думает» в
  первую очередь о подсолнечнике.
- 2020 — это «дальше всего от среднего» год. Cropwise там реально мажет
  на 4 t/ha. Если мы посмотрим, что NDVI показал в 2020 на этих полях,
  возможно, мы сможем построить модель, которая в таких аномальных годах
  систематически выигрывает.
- yield_maps: 26 полей × 4-8 maps = **карты реального урожая с комбайна**.
  Это гораздо чище, чем `productivity` (которое часто оценочное / нормированное),
  и может быть использовано как target и как ground-truth для дисперсии
  внутри поля.

## Список немедленных задач (если идём по C)

1. ~~Build root cause report~~ (готов).
2. Связать `field_scout_reports` с `(field_id, year, crop)` и собрать:
   - max BBCH к 1 июля / 1 августа / 1 сентября,
   - n threats events до as-of даты,
   - n disease events, n insect events.
3. Связать `soil_test_samples` с `(field_id, year)` и собрать
   per-field-per-year агрохимию.
4. Связать `yield_maps` с history_items и проверить:
   нет ли у нас полей, где `productivity` отличается от
   yield_maps `external_average` больше чем на 1 t/ha.
5. Решить, тянуть ли NDVI для остальных 513 полей.
