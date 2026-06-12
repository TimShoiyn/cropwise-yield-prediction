# ML Pipeline для прогноза урожайности

Полный пайплайн для построения ML-модели прогноза урожайности на основе данных Cropwise.

## Структура проекта

```
.
├── fetch_cropwise_data.py      # Выгрузка базовых данных из Operations API
├── fetch_ndvi_timeseries.py    # Выгрузка NDVI из Open Platform API
├── build_ml_dataset.py         # Сборка финального датасета
├── train_baseline_model.py     # Обучение baseline модели
├── data_raw/                   # Сырые данные
│   ├── fields.csv
│   ├── operations.csv
│   ├── crops.csv
│   ├── productivity_estimates.csv
│   ├── yield_maps.csv
│   └── ndvi_timeseries.csv     # (опционально, если есть доступ)
├── data_processed/             # Обработанные данные
│   └── ml_dataset.csv
└── models/                     # Обученные модели
    ├── baseline_model.pkl
    ├── feature_importance.png
    └── predictions_plot.png
```

## Шаг 1: Выгрузка базовых данных

Если еще не выгрузил базовые данные:

```powershell
$env:CROPWISE_API_KEY = "твой_токен"
python fetch_cropwise_data.py
```

Это создаст:
- `data_raw/fields.csv`
- `data_raw/operations.csv`
- `data_raw/crops.csv`
- `data_raw/productivity_estimates.csv`
- `data_raw/yield_maps.csv`

## Шаг 2: Выгрузка NDVI (опционально, но критично для качества)

**ВАЖНО:** NDVI данные находятся в другом API (Open Platform), который требует отдельного доступа.

```powershell
# Попробуй сначала Operations токен
$env:CROPWISE_OPEN_PLATFORM_TOKEN = $env:CROPWISE_API_KEY
python fetch_ndvi_timeseries.py

# Или если есть отдельный токен для Open Platform:
$env:CROPWISE_OPEN_PLATFORM_TOKEN = "отдельный_токен"
python fetch_ndvi_timeseries.py
```

### Если получаешь 401/403 ошибку:

Твой токен не имеет доступа к Remote Sensing API. Нужно:

1. Связаться с заказчиком, который дал токен
2. Запросить активацию **Remote Sensing contract** для твоего аккаунта
3. Или попросить отдельный OAuth токен для Open Platform API

**Без NDVI модель будет работать, но качество будет низким (R² ~0.3-0.5).**

## Шаг 3: Сборка ML датасета

Объединяет все данные в финальный датасет:

```powershell
python build_ml_dataset.py
```

Создает:
- `data_processed/ml_dataset.csv` - финальный датасет с фичами и таргетом

### Что включает датасет:

**Базовые фичи:**
- Площадь поля, координаты
- Количество операций по типам
- NPK (азот, фосфор, калий) из операций
- Длительность сезона
- Культура

**NDVI фичи (если доступны):**
- Средний/мин/макс NDVI за сезон
- NDVI по фазам (ранний/средний/поздний сезон)
- Тренд NDVI
- Количество наблюдений

**Таргет:**
- `target_yield_t_ha` - урожайность в т/га из `productivity_estimates`

## Шаг 4: Обучение модели

```powershell
python train_baseline_model.py
```

Создает:
- `models/baseline_model.pkl` - обученная модель
- `models/feature_importance.png` - график важности фич
- `models/predictions_plot.png` - график предсказаний vs реальные значения

### Метрики модели:

- **R² (коэффициент детерминации)** - чем ближе к 1, тем лучше
- **RMSE (Root Mean Squared Error)** - ошибка в т/га
- **MAE (Mean Absolute Error)** - средняя абсолютная ошибка в т/га

### Интерпретация качества:

- **R² < 0.3** - низкое качество, нужны дополнительные данные (особенно NDVI)
- **0.3 ≤ R² < 0.5** - среднее качество, можно улучшить добавлением NDVI
- **R² ≥ 0.5** - хорошее качество

## Зависимости

Установи все зависимости:

```powershell
pip install -r requirements.txt
```

## Проблемы и решения

### Проблема: NDVI API возвращает 401/403

**Решение:** Запроси у заказчика доступ к Remote Sensing API или отдельный токен.

### Проблема: Низкое качество модели (R² < 0.3)

**Возможные причины:**
1. Нет NDVI данных (критично!)
2. Недостаточно данных
3. Нужна дополнительная feature engineering

**Решения:**
1. Получить доступ к NDVI
2. Добавить метеоданные
3. Увеличить объем данных
4. Попробовать другие алгоритмы (XGBoost, Random Forest)

### Проблема: Много пропусков в данных

Скрипт автоматически заполняет пропуски медианой для числовых фич. Если пропусков слишком много - проверь исходные данные.

## Debug sanity checks

Если нужно быстро проверить, что датасет выглядит адекватно (NaN, таргет по годам, фильтр культур, базовая “санити” модель на случайном KFold), запусти:

```powershell
python debug_dataset_sanity.py
```

Артефакты сохраняются в `reports/debug/` (консольные логи, списки фич, NaN-таблицы, гистограммы таргета, размах таргета по годам).

## Следующие шаги

После получения NDVI данных:

1. Перезапусти `fetch_ndvi_timeseries.py`
2. Пересобери датасет: `build_ml_dataset.py`
3. Переобучи модель: `train_baseline_model.py`

Ожидаемое улучшение качества: R² должен вырасти с ~0.3-0.4 до ~0.6-0.8.

## Для встречи 10 февраля

**Что показать:**

1. ✅ Инвентаризация данных (что есть, чего не хватает)
2. ✅ Структура датасета (какие фичи)
3. ⚠️ Блокер: NDVI недоступен → нужен Remote Sensing contract
4. 🎯 Baseline модель без NDVI (покажет R²~0.3-0.5)
5. 📋 План после получения NDVI

**Что запросить:**

- Активацию Remote Sensing API для твоего токена
- Или экспорт NDVI time series за 2021-2022 в CSV
- Подтверждение: какие годы критичны для модели
