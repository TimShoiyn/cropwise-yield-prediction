# v23 Cropwise full: финальный результат

## Что изменилось

v23 впервые использует полный новый pull из Cropwise API: не 180 строк старого локального датасета и не peer-прокси, а реальные поля из аккаунта.

- dataset: **1 903** строгих target-строки;
- поля: **483** поля с валидным target;
- годы: **2017-2025**;
- спутник: `ndvi`;
- дополнительные ряды Cropwise: `temperature`, `soil_moisture`;
- soil tests: почвенные анализы, leakage-safe latest before as-of;
- validation: walk-forward by year, без использования Cropwise как feature.

Важно: `historical_values` не отдаёт NDRE/EVI/NDMI. API игнорирует `type` и возвращает фактически `ndvi`, `temperature`, `soil_moisture` и source-specific NDVI (`ndvi_s2a/s2b/...`).

## Модели

- `v23_full` — CatBoost с `field_id` как categorical feature.
- `v23_no_field` — CatBoost без `field_id`, честнее для переноса на новые поля.
- `cropwise` — внешний benchmark, в признаки не попадает.

## Основная таблица (`target >= 1 т/га`)

| scenario       | asof   | model        |    n |   n_target_ge_1 |   mape_target_ge_1 |   r2_target_ge_1 |    r2 |   wcy_r2 |
|:---------------|:-------|:-------------|-----:|----------------:|-------------------:|-----------------:|------:|---------:|
| all_crops      | 1 Jul  | v23_full     | 1706 |            1531 |             30.229 |            0.17  | 0.196 |    0.093 |
| all_crops      | 1 Jul  | v23_no_field | 1706 |            1531 |             30.503 |            0.17  | 0.196 |    0.09  |
| all_crops      | 1 Jul  | cropwise     | 1667 |            1497 |             28.74  |            0.215 | 0.262 |    0.052 |
| all_crops      | 1 Aug  | v23_full     | 1706 |            1531 |             30.875 |            0.168 | 0.197 |    0.04  |
| all_crops      | 1 Aug  | v23_no_field | 1706 |            1531 |             31.813 |            0.13  | 0.163 |   -0.017 |
| all_crops      | 1 Aug  | cropwise     | 1667 |            1497 |             27.355 |            0.289 | 0.32  |    0.089 |
| all_crops      | 1 Sep  | v23_full     | 1706 |            1531 |             31.186 |            0.119 | 0.187 |    0.049 |
| all_crops      | 1 Sep  | v23_no_field | 1706 |            1531 |             31.011 |            0.15  | 0.216 |    0.078 |
| all_crops      | 1 Sep  | cropwise     | 1667 |            1497 |             27.08  |            0.291 | 0.334 |    0.098 |
| wheat_combined | 1 Jul  | v23_full     |  791 |             719 |             28.798 |            0.137 | 0.158 |    0.058 |
| wheat_combined | 1 Jul  | v23_no_field |  791 |             719 |             28.999 |            0.13  | 0.159 |    0.05  |
| wheat_combined | 1 Jul  | cropwise     |  769 |             698 |             28.273 |            0.153 | 0.214 |    0.028 |
| wheat_combined | 1 Aug  | v23_full     |  791 |             719 |             28.71  |            0.127 | 0.144 |    0.043 |
| wheat_combined | 1 Aug  | v23_no_field |  791 |             719 |             29.133 |            0.108 | 0.148 |    0.048 |
| wheat_combined | 1 Aug  | cropwise     |  769 |             698 |             27.463 |            0.201 | 0.255 |    0.057 |
| wheat_combined | 1 Sep  | v23_full     |  791 |             719 |             30.914 |            0.159 | 0.172 |    0.066 |
| wheat_combined | 1 Sep  | v23_no_field |  791 |             719 |             30.935 |            0.131 | 0.175 |    0.08  |
| wheat_combined | 1 Sep  | cropwise     |  769 |             698 |             28.444 |            0.188 | 0.242 |    0.037 |
| sunflower      | 1 Jul  | v23_full     |  403 |             372 |             24.605 |            0.048 | 0.169 |    0.132 |
| sunflower      | 1 Jul  | v23_no_field |  403 |             372 |             24.591 |            0.062 | 0.173 |    0.118 |
| sunflower      | 1 Jul  | cropwise     |  397 |             367 |             26.508 |           -0.229 | 0.046 |    0.021 |
| sunflower      | 1 Aug  | v23_full     |  403 |             372 |             23.039 |            0.135 | 0.259 |    0.175 |
| sunflower      | 1 Aug  | v23_no_field |  403 |             372 |             22.992 |            0.137 | 0.268 |    0.179 |
| sunflower      | 1 Aug  | cropwise     |  397 |             367 |             23.399 |            0.106 | 0.255 |    0.145 |
| sunflower      | 1 Sep  | v23_full     |  403 |             372 |             23.558 |            0.082 | 0.239 |    0.146 |
| sunflower      | 1 Sep  | v23_no_field |  403 |             372 |             23.275 |            0.109 | 0.26  |    0.151 |
| sunflower      | 1 Sep  | cropwise     |  397 |             367 |             21.158 |            0.247 | 0.363 |    0.237 |

## Короткий честный вывод

**v23 стал научно намного сильнее как dataset/pipeline, но пока не стал стабильным победителем Cropwise.**

По всем культурам Cropwise всё ещё лучше по pooled R² и MAPE: примерно **27-29% MAPE** против **30-32%** у v23. Это значит, что Cropwise лучше ловит общую межгодовую/межкультурную шкалу урожайности.

По пшенице v23 уже близко на 1 июля: **28.8-29.0% MAPE** против **28.3%** у Cropwise. По R² Cropwise выше, но по `within_crop_year_R2` v23 иногда лучше, особенно на 1 сентября: `v23_no_field = 0.080` против `Cropwise = 0.037`. Это значит: внутри одного crop-year v23 иногда лучше ранжирует поля, даже если хуже угадывает абсолютный уровень.

По подсолнечнику v23 реально сильный на ранних датах: на **1 июля** v23 лучше Cropwise по нормальному MAPE (**24.6% vs 26.5%**) и сильно лучше по R² / field ranking. На **1 августа** v23 тоже чуть лучше по MAPE (**23.0% vs 23.4%**) и лучше по `within_crop_year_R2`. На **1 сентября** Cropwise снова выигрывает.

## Что это значит для диссертации

Теперь можно честно сказать: после получения полного API-доступа мы сняли главный data bottleneck и построили модель на реальных 483 полях / 1903 field-year наблюдениях. Модель уже конкурентна Cropwise по отдельным культурам и датам, особенно по раннему подсолнечнику, но для полной победы нужно улучшать target quality / crop-specific calibration, а не просто добавлять ещё CatBoost-фичи.

## Следующий разумный шаг

v24 должен быть не “ещё одна общая модель”, а crop-specific calibration:

- отдельные модели/калибровки для wheat и sunflower;
- обучение loss/weights на `target >= 1`, чтобы низкоурожайные аномалии не ломали MAPE;
- residual model поверх v23 + Cropwise benchmark только как отдельный сценарий, если нужно практическое улучшение, но не как основной независимый thesis claim;
- отдельный target audit для нулевых/низких урожайностей, потому что они дают MAPE 100%+ и искажают all-crops вывод.
