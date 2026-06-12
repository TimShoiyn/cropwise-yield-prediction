# v29 — pruned management feature ablation

Дата: 2026-06-11.

Цель: проверить, можно ли заменить полный `v28_all_mgmt` компактным набором устойчивых management-фич.
Проверка честная: одинаковые S2-covered + pre-harvest строки, без `field_id`, walk-forward по годам.

## Что оставили в pruned-наборе

- `mgmt_application_count_asof`
- `mgmt_days_since_last_application_asof`
- `mgmt_n_kg_ha_asof`
- `mgmt_p2o5_kg_ha_asof`
- `mgmt_s_kg_ha_asof`
- `mgmt_herbicide_rate_asof`
- `mgmt_fungicide_rate_asof`
- `mgmt_insecticide_rate_asof`

## Короткий вывод

Pruned-набор **не победил универсально**. Мой изначальный prune оказался слишком агрессивным для wheat/all_crops:

- `all_crops 1 Aug`: полный v28 лучше — R² **0.119** против v29 **-0.132**;
- `wheat 1 Aug`: полный v28 лучше — R² **0.299** против v29 **0.268**;
- `sunflower 1 Aug`: pruned лучше — MAPE **21.8%**, R² **0.217** (лучше v27/v28 и лучше Cropwise-asof по MAPE);
- `barley`: management вреден, лучше оставить v27 без management.

## Практическая policy

В финальную систему не надо тащить один одинаковый набор для всех культур. Лучшее правило сейчас:

| Сценарий | Лучшая версия | Почему |
|---|---|---|
| all_crops 1 Aug | `v28_all_mgmt` | полный менеджмент даёт сильный прирост против v27 |
| wheat 1 Aug | `v28_all_mgmt` | лучший R²/MAPE среди наших вариантов |
| sunflower 1 Aug | `v29_pruned_mgmt` | лучший MAPE/R², лучше Cropwise-asof по MAPE |
| barley | `v27_s2` или без management | management шумит на малой выборке |

Это хороший научный результат: management-фичи полезны, но **культуро-специфично**, а не как универсальная добавка.

## Полная таблица

| scenario       | asof   | model           |   n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:-------|:----------------|----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | 1 Jul  | v27_s2          | 946 | 42.291 |  0.155 |    0.141 |  1.095 | 0.901 |
| all_crops      | 1 Jul  | v28_all_mgmt    | 946 | 41.895 |  0.15  |    0.146 |  1.098 | 0.898 |
| all_crops      | 1 Jul  | v29_pruned_mgmt | 946 | 43.923 |  0.083 |    0.107 |  1.141 | 0.933 |
| all_crops      | 1 Jul  | cropwise_asof   | 943 | 29.348 |  0.333 |    0.025 |  0.972 | 0.751 |
| all_crops      | 1 Aug  | v27_s2          | 904 | 46.976 | -0.051 |    0.04  |  1.175 | 0.966 |
| all_crops      | 1 Aug  | v28_all_mgmt    | 904 | 41.45  |  0.119 |    0.156 |  1.076 | 0.878 |
| all_crops      | 1 Aug  | v29_pruned_mgmt | 904 | 48.395 | -0.132 |    0.031 |  1.22  | 0.999 |
| all_crops      | 1 Aug  | cropwise_asof   | 901 | 27.829 |  0.44  |    0.18  |  0.857 | 0.666 |
| all_crops      | 1 Sep  | v27_s2          |  63 | 22.123 |  0.146 |    0.149 |  0.762 | 0.611 |
| all_crops      | 1 Sep  | v28_all_mgmt    |  63 | 21.939 |  0.168 |    0.171 |  0.752 | 0.592 |
| all_crops      | 1 Sep  | v29_pruned_mgmt |  63 | 21.659 |  0.238 |    0.166 |  0.72  | 0.567 |
| all_crops      | 1 Sep  | cropwise_asof   |  63 | 24.713 |  0.083 |    0.068 |  0.79  | 0.629 |
| wheat_combined | 1 Jul  | v27_s2          | 490 | 33.801 |  0.308 |    0.124 |  1.035 | 0.835 |
| wheat_combined | 1 Jul  | v28_all_mgmt    | 490 | 37.168 |  0.211 |    0.108 |  1.104 | 0.906 |
| wheat_combined | 1 Jul  | v29_pruned_mgmt | 490 | 33.673 |  0.281 |    0.131 |  1.054 | 0.848 |
| wheat_combined | 1 Jul  | cropwise_asof   | 490 | 28.367 |  0.338 |    0.014 |  1.012 | 0.772 |
| wheat_combined | 1 Aug  | v27_s2          | 456 | 38.495 |  0.194 |    0.153 |  1.047 | 0.877 |
| wheat_combined | 1 Aug  | v28_all_mgmt    | 456 | 33.761 |  0.299 |    0.142 |  0.977 | 0.804 |
| wheat_combined | 1 Aug  | v29_pruned_mgmt | 456 | 35.25  |  0.268 |    0.125 |  0.998 | 0.828 |
| wheat_combined | 1 Aug  | cropwise_asof   | 456 | 26.44  |  0.454 |    0.198 |  0.862 | 0.676 |
| sunflower      | 1 Jul  | v27_s2          | 210 | 24.071 |  0.033 |    0.077 |  0.743 | 0.572 |
| sunflower      | 1 Jul  | v28_all_mgmt    | 210 | 23.972 |  0.075 |    0.106 |  0.727 | 0.557 |
| sunflower      | 1 Jul  | v29_pruned_mgmt | 210 | 23.884 |  0.077 |    0.104 |  0.726 | 0.558 |
| sunflower      | 1 Jul  | cropwise_asof   | 210 | 25.657 | -0.335 |    0.003 |  0.873 | 0.71  |
| sunflower      | 1 Aug  | v27_s2          | 209 | 22.604 |  0.179 |    0.147 |  0.686 | 0.526 |
| sunflower      | 1 Aug  | v28_all_mgmt    | 209 | 22.606 |  0.18  |    0.141 |  0.686 | 0.526 |
| sunflower      | 1 Aug  | v29_pruned_mgmt | 209 | 21.817 |  0.217 |    0.182 |  0.67  | 0.508 |
| sunflower      | 1 Aug  | cropwise_asof   | 209 | 23.653 | -0.029 |    0.111 |  0.768 | 0.611 |
| barley         | 1 Jul  | v27_s2          |  83 | 39.374 |  0.083 |    0.289 |  1.355 | 1.149 |
| barley         | 1 Jul  | v28_all_mgmt    |  83 | 41.162 |  0.003 |    0.252 |  1.413 | 1.18  |
| barley         | 1 Jul  | v29_pruned_mgmt |  83 | 40.715 |  0.008 |    0.253 |  1.409 | 1.184 |
| barley         | 1 Jul  | cropwise_asof   |  83 | 42.438 |  0.106 |    0.116 |  1.337 | 1.109 |
| barley         | 1 Aug  | v27_s2          |  84 | 40.36  | -0.007 |    0.161 |  1.425 | 1.195 |
| barley         | 1 Aug  | v28_all_mgmt    |  84 | 43.552 | -0.231 |    0.171 |  1.575 | 1.342 |
| barley         | 1 Aug  | v29_pruned_mgmt |  84 | 42.921 | -0.202 |    0.193 |  1.556 | 1.326 |
| barley         | 1 Aug  | cropwise_asof   |  84 | 40.149 |  0.242 |    0.245 |  1.236 | 0.986 |

## Следующий шаг

Сделать `v30_policy`: автоматически выбирать лучшую ветку по сценарию/культуре:

- wheat/all_crops → v28 all management;
- sunflower → v29 pruned management;
- barley → v27 S2 only;
- 1 Aug как основная дата.

После этого можно оформлять финальный профессорский отчёт с честной линией:
S2 помогает, management помогает выборочно, Cropwise пока сильнее на wheat/all_crops, но sunflower мы бьём по MAPE.
