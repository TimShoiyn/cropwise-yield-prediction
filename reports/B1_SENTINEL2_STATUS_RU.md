# B1 Sentinel-2 red-edge/EVI/GCVI status

Цель B1: добавить признаки, которых нет в Cropwise API:

- `NDRE` — red-edge индекс, важен для плотного полога и поздних стадий;
- `EVI` — устойчивее NDVI при насыщении;
- `GCVI` — chlorophyll/green canopy signal;
- `NDVI` из Sentinel-2 — sanity check против Cropwise NDVI.

## Что уже получилось

Рабочий источник найден: **Microsoft Planetary Computer STAC**, Sentinel-2 L2A.
Google Earth Engine токен не нужен.

Smoke-test успешен:

- поле `195`;
- окно июль 2023;
- найдено 10 Sentinel-2 сцен;
- посчитаны `NDVI`, `NDRE`, `EVI`, `GCVI` по полигону поля;
- файл: `reports/microscope/sentinel2_indices_smoke.csv`.

Пример результата:

```text
field_id=195
date=2023-07-13
cloud_cover≈0
ndvi=0.286
ndre=0.217
evi=0.302
gcvi=0.863
```

## Что реализовано

Созданы скрипты:

- `scripts/external/sentinel2_indices_smoke.py`
- `scripts/external/fetch_sentinel2_indices_b1.py`

Batch extractor:

- берёт v24 normal-yield rows (`target >= 1`);
- для каждого `(field_id, year, asof_tag)` ищет лучший Sentinel-2 снимок за 30 дней до даты прогноза;
- считает:
  - `s2_ndvi_mean`;
  - `s2_ndre_mean`;
  - `s2_evi_mean`;
  - `s2_gcvi_mean`;
  - pixel counts и cloud cover;
- пишет в `data_raw/sentinel2_indices_b1.csv`;
- **resumable**: если оборвётся, повторный запуск продолжит с непройденных ключей.

## Текущий прогон

Запущен полный прогон:

```bash
python3 scripts/external/fetch_sentinel2_indices_b1.py --cloud-lt 35
```

Перед полным прогоном sample `--limit 20` прошёл успешно:

- 20/20 rows `status=ok`;
- первые значения выглядят физически правдоподобно;
- скорость примерно 4-5 секунд на `(field, year, asof)`.

Ожидаемый объём полного B1:

- примерно `1684 normal-yield rows × 3 as-of = 5052` задач;
- время может быть несколько часов.

## Технические детали

Индексы:

- `NDVI = (B08 - B04) / (B08 + B04)` at 10m;
- `EVI = 2.5*(B08-B04)/(B08 + 6*B04 - 7.5*B02 + 1)` at 10m;
- `GCVI = B08/B03 - 1` at 10m;
- `NDRE = (B8A - B05)/(B8A + B05)` at 20m.

Почему NDRE отдельно: red-edge band `B05` имеет 20m resolution, а обычные RGB/NIR
для EVI/GCVI — 10m. Нельзя смешивать их в одном массиве без ресэмплинга, поэтому
NDRE считается на паре `B8A/B05`.

## Ограничения текущей версии

1. Пока используется **один лучший снимок за 30 дней до as-of**, а не полный временной ряд.
   Это быстрый и реалистичный первый шаг.
2. Cloud filtering идёт по scene-level `eo:cloud_cover`; per-pixel cloud/shadow mask можно
   улучшить позже через SCL resampling.
3. Полный прогон долгий, но resumable.

## Что делать после завершения выгрузки

1. Проверить coverage:

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('data_raw/sentinel2_indices_b1.csv')
print(df.status.value_counts())
print(df.groupby('asof_tag').status.value_counts())
PY
```

2. Смёрджить `s2_*` в v24 датасеты.
3. Запустить v25:

- `v25_s2_full`;
- `v25_s2_pruned`;
- по культурам wheat/sunflower;
- честно на `target >= 1`;
- сравнение с Cropwise на тех же строках.

Ожидаемый эффект: больше всего B1 должен помочь **пшенице и 1 Sep**, где NDVI
насыщается и Cropwise пока сильнее.
