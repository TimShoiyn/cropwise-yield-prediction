# Прозрачная картина текущей модели (что изменилось, что нет, и что дальше)

Этот файл отвечает на главный вопрос:

> “Мы действительно поменяли только сезонность NDVI/погоды, а всё остальное осталось как раньше?”

Короткий ответ:

- **Да, в `cropwindow_v1` основное изменение — это сезонные окна и состав NDVI/weather признаков.**
- **Нет, это не единственная модель проекта:** ранее тестировались и модели с `ops_*` (операции/удобрения), но в текущей финальной конфигурации cropwindow эти признаки не используются.

---

## 1) Что сейчас считается “текущей” моделью

Текущий фокус (финальная конфигурация для презентации):

- `dataset_name = full_extended_cropwindow_v1_allcrops`
- `model_type = all_no_leak_reg_cropwindow_allcrops`
- Алгоритм: `GradientBoostingRegressor` (регуляризованный)
- Валидация: `GroupKFold(n_splits=5)` по `year`

Файл метрик:
- `models/models_comparison.csv` (строка `full_extended_cropwindow_v1_allcrops`)

---

## 2) Что именно поменялось относительно “старой” full_extended

Сравниваем:

- **Старая база:** `data_processed/ml_dataset_full_extended.csv`
- **Новая версия:** `data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv`

### 2.1 NDVI

В `cropwindow_v1` оставлены компактные сезонные NDVI-признаки:

- `ndvi_observations`
- `ndvi_mean_season`
- `ndvi_max_season`
- `ndvi_early`, `ndvi_mid`, `ndvi_late`

В старом `full_extended` были дополнительные “legacy” NDVI:

- `ndvi_min_season`, `ndvi_std_season`, `ndvi_trend`
- `ndvi_green_period_days`, `ndvi_peak_date`, `ndvi_peak_doy`, `ndvi_integral`

### 2.2 Погода

В `cropwindow_v1` используется 5 сезонных погодных признаков:

- `weather_temp_avg_season`
- `weather_gdd_season`
- `weather_precip_sum_season`
- `weather_precip_sum_early`
- `weather_hot_days`

В старом `full_extended` использовались другие агрегаты (`weather_temp_*`, `weather_precip_*`, `weather_snow_*`).

### 2.3 Сезонность (главное)

Погода и NDVI считаются в окне, зависящем от культуры:

- `wheat_spring`: май–сентябрь
- `sunflower`: май–октябрь
- прочие: дефолт (май–октябрь)

То есть **ключевое изменение действительно сезонность + новый набор агрегатов NDVI/погоды**.

---

## 3) Что осталось прежним (не менялось концептуально)

Остались:

- единица наблюдения: `field_id × year`
- таргет: `target_yield_t_ha`
- культура: `crop_id`, `prev_crop_id`
- геометрия/координаты: `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`
- почва: `field_soil_*` (последний тест по полю)
- честная валидация: GroupKFold по годам

То есть модель по-прежнему прогнозирует урожайность поля за год, но на другом (более “агрономическом”) наборе сезонных признаков.

---

## 4) Какие признаки реально входят в текущую финальную модель

Текущий датасет `ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv` содержит 29 колонок:

- ключи/таргет: `field_id`, `year`, `target_yield_t_ha`
- культура: `crop_id`, `prev_crop_id`
- геометрия: `field_tillable_area`, `field_calculated_area`, `field_lat`, `field_long`
- NDVI: `ndvi_observations`, `ndvi_mean_season`, `ndvi_max_season`, `ndvi_early`, `ndvi_mid`, `ndvi_late`
- почва: `field_soil_pH`, `field_soil_OM`, `field_soil_P`, `field_soil_K`, `field_soil_N`, `field_soil_N_NO3`, `field_soil_Mg`, `field_soil_CEC`, `field_soil_Ca_saturation`
- погода: `weather_temp_avg_season`, `weather_gdd_season`, `weather_precip_sum_season`, `weather_precip_sum_early`, `weather_hot_days`

### 4.1 Расшифровка каждого показателя (простыми словами)

Общее для всех признаков ниже:

- период текущего датасета: **2010–2025**;
- одна строка = одно `field_id × year`;
- значит каждый признак даёт **одно значение на строку** (максимум ~441 значений по всему датасету, часть может быть NaN).

#### Ключи и цель

- `field_id` — ID поля (служебный ключ, не должен использоваться как фича в модели).
- `year` — год сезона (служебный ключ и группировка для GroupKFold).
- `target_yield_t_ha` — целевая урожайность, которую предсказывает модель (т/га).

#### Культура

- `crop_id` — код текущей культуры на поле в этом году; помогает модели понимать, что потенциал урожая у разных культур разный.
- `prev_crop_id` — код предшественника (какая культура была в прошлом году); это признак севооборота, влияет на плодородие/стресс/фон поля.

#### Геометрия поля

- `field_tillable_area` — обрабатываемая площадь поля (обычно га); косвенно описывает размер и агротехнику.
- `field_calculated_area` — площадь по геометрии контура (обычно га); служит уточнением/контролем площади.
- `field_lat` — широта центра поля; отражает региональные климатические различия.
- `field_long` — долгота центра поля; вместе с `field_lat` фиксирует пространственный контекст.

#### NDVI (считаются внутри сезонного окна культуры)

- `ndvi_observations` — сколько NDVI-точек попало в окно; это индикатор надёжности агрегатов.
- `ndvi_mean_season` — средний NDVI за окно; общий “уровень вегетации”.
- `ndvi_max_season` — максимальный NDVI за окно; пик биомассы/развития.
- `ndvi_early` — средний NDVI в ранней трети окна; старт/всходы.
- `ndvi_mid` — средний NDVI в средней трети окна; активная вегетация.
- `ndvi_late` — средний NDVI в поздней трети окна; фаза завершения/созревания.

#### Почва (берётся последний доступный SoilTest по полю)

- `field_soil_pH` — кислотность почвы.
- `field_soil_OM` — органическое вещество.
- `field_soil_P` — обеспеченность фосфором.
- `field_soil_K` — обеспеченность калием.
- `field_soil_N` — обеспеченность азотом.
- `field_soil_N_NO3` — нитратная форма азота (если есть).
- `field_soil_Mg` — магний.
- `field_soil_CEC` — катионно-обменная ёмкость.
- `field_soil_Ca_saturation` — насыщенность кальцием.

Важно:

- `field_soil_CEC` и `field_soil_Ca_saturation` сейчас в выгрузке 100% пустые, поэтому автоматически удаляются перед обучением.

#### Погода (daily -> seasonal aggregates в окне культуры)

- `weather_temp_avg_season` — средняя температура за окно (°C).
- `weather_gdd_season` — сумма активных температур (GDD) за окно.
- `weather_precip_sum_season` — сумма осадков за окно.
- `weather_precip_sum_early` — осадки в первой половине окна.
- `weather_hot_days` — число жарких дней в окне (`temp_max > 30°C`).

Как модель использует погодные признаки:

- если по строке погода есть -> признаки напрямую участвуют в прогнозе;
- если погоды нет -> значение импрютируется в Pipeline, и модель опирается сильнее на NDVI/культуру/геометрию;
- доля строк без всей погоды (`all_weather_nan_share`) сейчас:
  - all: ~0.186
  - sunflower: ~0.287
  - wheat: 0.000

### Важная техническая деталь

Перед обучением автоматически удаляются признаки, где 100% NaN:

- `field_soil_CEC`
- `field_soil_Ca_saturation`

Это видно в логах обучения:

- `Dropping all-NaN feature columns (2): ['field_soil_CEC', 'field_soil_Ca_saturation']`

---

## 5) Что **не входит** в текущую cropwindow финальную модель

В этой конфигурации **нет** `ops_*` признаков:

- `ops_count_*`
- `ops_npk_*`
- `ops_yield_t_ha`
- `ops_days_*`

Поэтому если на слайде написано “учтены операции/удобрения” для именно `full_extended_cropwindow_v1_allcrops`, это будет неточно.

Корректно:

> Операционные признаки тестировались отдельно на других датасетах (`ml_dataset_with_ndvi`, `ml_dataset_ops_ndvi_2021_2025`), но в финальной cropwindow-конфигурации не используются.

---

## 6) Были ли вообще протестированы операции/удобрения/предшественник?

Да, но по-разному:

- `prev_crop_id` — **входит** в текущую cropwindow модель.
- `ops_*` и `ops_npk_*` — тестировались в других датасетах:
  - `data_processed/ml_dataset_with_ndvi.csv`
  - `data_processed/ml_dataset_ops_ndvi_2021_2025.csv`

Почему не в финальной cropwindow:

- финальный набор выбирался по CV-метрикам и устойчивости;
- ops-датасет маленький и сильно переобучается (очень высокий train, плохой CV).

---

## 7) Текущие результаты (строго по файлу `models/models_comparison.csv`)

### 7.1 Baseline на cropwindow (без спец-фильтра по культурам)

`full_extended_cropwindow_v1 / all_no_leak`:

- CV R² = **0.2804 ± 0.2393**
- CV RMSE = **5.5926 ± 1.4812** т/га
- CV MAPE = **17.9229 ± 3.8069** %
- Train R² = **0.9901**

### 7.2 Регуляризованный вариант (allcrops / wheat / sunflower)

`full_extended_cropwindow_v1_allcrops / all_no_leak_reg_cropwindow_allcrops`:

- CV R² = **0.3094 ± 0.2681**
- CV RMSE = **5.3893 ± 1.2889** т/га
- CV MAPE = **17.9070 ± 3.9550** %
- Train R² = **0.9462**

`full_extended_cropwindow_v1_sunflower / all_no_leak_reg_cropwindow_sunflower`:

- CV R² = **0.3267 ± 0.1528**
- CV RMSE = **3.8404 ± 0.3115** т/га
- CV MAPE = **14.9029 ± 1.1382** %
- Train R² = **0.9607**

`full_extended_cropwindow_v1_wheat / all_no_leak_reg_cropwindow_wheat`:

- CV R² = **-0.2284 ± 0.6535**
- CV RMSE = **4.9732 ± 1.9844** т/га
- CV MAPE = **16.8840 ± 11.5690** %
- Train R² = **0.9970**

---

## 8) Почему у wheat отрицательный CV R² (и это не “поломка”)

Из `reports/debug/target_by_year_wheat.csv`:

- некоторые годы очень маленькие: `n=2`, `n=3`, `n=2`, `n=5`

При GroupKFold по годам это приводит к:

- нестабильности фолдовых метрик,
- большому std по R² (`±0.6535`),
- и возможному отрицательному среднему R², даже если RMSE/MAPE выглядят приемлемо.

Это не означает, что “в данных нет сигнала”:

- sanity random KFold показывает для wheat `simpleKFold_R2 ~ 0.742`

Значит проблема не в полном отсутствии сигнала, а в малом N + жестком разбиении по годам.

---

## 9) Ограничения данных прямо сейчас (честно)

1. В части строк отсутствует вся погода:
   - all: `all_weather_nan_share = 0.186`
   - sunflower: `0.287`
   - wheat: `0.000`

2. Есть выбросы таргета:
   - в all диапазон `6.08 ... 64.86` т/га

3. По wheat мало наблюдений и мало лет для устойчивого GroupKFold.

4. Некоторые soil-показатели полностью пустые (`CEC`, `Ca_saturation`) и исключаются.

---

## 10) Можно ли улучшить модель **без новых данных**?

Да, можно сделать 3 практичных шага:

1. **Стабилизировать валидацию для wheat**
   - добавлять в отчёт RMSE/MAPE как основные метрики для малых подгрупп;
   - считать доверительный интервал/бутстрэп для ошибки.

2. **Улучшить обработку выбросов**
   - протестировать робастный таргет-трансформ (например, clip/winsorize или log1p на эксперименте).

3. **Фичи из того, что уже есть**
   - добавить взаимодействия на уровне признаков (например, `ndvi_mid × weather_gdd_season`, `ndvi_early × weather_precip_sum_early`);
   - аккуратно сравнить с CatBoost в той же схеме GroupKFold.

---

## 11) Формулировка для слайда (готовый текст)

> В текущей версии модели `cropwindow_v1` мы изменили прежде всего сезонную логику признаков: NDVI и погода считаются в окнах, зависящих от культуры (для яровой пшеницы май–сентябрь, для подсолнечника май–октябрь). Геометрия поля, культура/предшественник и почвенные признаки сохранены.  
> Операционные признаки (`ops_*`, включая NPK) были протестированы в отдельных экспериментах, но в финальную cropwindow-конфигурацию не вошли.  
> Текущая лучшая конфигурация на всех культурах даёт CV \(R^2\) ≈ 0.31 и MAPE ≈ 17.9% при GroupKFold по годам. Для подсолнечника качество выше, для пшеницы \(R^2\) нестабилен из-за малого числа наблюдений по годам.

---

## 12) Какие файлы показать

- Метрики:
  - `models/models_comparison.csv`
- OOF-графики:
  - `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_allcrops_oof.png`
  - `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_sunflower_oof.png`
  - `models/predictions_plot_full_extended_cropwindow_v1_all_no_leak_reg_cropwindow_wheat_oof.png`
- Санити:
  - `reports/debug/summary.txt`
  - `reports/debug/target_by_year_wheat.csv`

