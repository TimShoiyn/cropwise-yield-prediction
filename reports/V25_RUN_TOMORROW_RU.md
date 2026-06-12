# v25: что запустить завтра (после B1)

Простыми словами: B1 качает со спутника Sentinel-2 три «продвинутых индекса
зелёности» (NDRE, EVI, GCVI), которых нет в Cropwise. Когда скачается — мы
вклеиваем их в данные и проверяем, стало ли лучше.

## Шаг 0. Проверить, докачался ли B1

```bash
pgrep -f fetch_sentinel2_indices_b1.py && echo "ещё качает" || echo "остановлено"

python3 - <<'PY'
import pandas as pd
d = pd.read_csv('data_raw/sentinel2_indices_b1.csv')
ok = (d.status=='ok').sum()
print('ok строк:', ok)
print('по датам:', d[d.status=='ok'].asof_tag.value_counts().to_dict())
print('статусы:', d.status.value_counts().to_dict())
PY
```

Полный объём ~5052. Если `ok` близко к этому и по всем трём датам (07_01, 08_01,
09_01) — готово.

## Шаг 1. Если остановилось, но не докачано — просто перезапустить

Скрипт resumable, продолжит с места:

```bash
python3 scripts/external/fetch_sentinel2_indices_b1.py --cloud-lt 35
```

## Шаг 2. Собрать v25 датасет (вклеить S2 в v24)

```bash
python3 scripts/asof/build_v25_s2_dataset.py
```

Создаст `data_processed/ml_dataset_v25_s2_asof_*.csv` и
`reports/V25_DATASET_SUMMARY.md`.

Можно запускать даже на частичной выгрузке — строки без снимка останутся пустыми,
модель это переживёт. Но для честной картины лучше дождаться полной.

## Шаг 3. Обучить и сравнить (главный момент)

```bash
python3 scripts/asof/evaluate_v25_s2.py
```

Это **честный ablation на одинаковых строках**:

- `v24_base` — наши признаки без спутниковых индексов;
- `v25_s2` — те же признаки + NDRE/EVI/GCVI;
- `cropwise` — бенчмарк на тех же строках.

Результат: `reports/V25_S2_FINAL_RU.md` +
`models_v2/asof_comparison/asof_results_v25_s2.csv`.

## На что смотреть

- Если `v25_s2` < `v24_base` по MAPE и/или выше по `wcy_r2` — значит red-edge
  реально помог (особенно ждём эффект на пшенице и на 1 сентября).
- Если разницы нет — значит для нашего региона NDVI почти всё уже объясняет, и
  тогда главный рычаг переходит к B2 (менеджмент/фосфор).

## Файлы, созданные сегодня

- `scripts/asof/build_v25_s2_dataset.py`
- `scripts/asof/evaluate_v25_s2.py`
- `scripts/asof/feature_sets_v24.py` (добавлены S2-признаки)
- `reports/V25_RUN_TOMORROW_RU.md` (этот файл)
