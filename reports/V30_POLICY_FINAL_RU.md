# v30 — финальная policy-модель (1 Aug)

Дата: 2026-06-11.

Основная дата: **1 августа** (лучший компромисс: до уборки, сезонный сигнал уже есть).

Policy выбирает лучшую ветку по сценарию/культуре:

| Сценарий | Выбранная ветка |
|---|---|
| `all_crops` | `v28_all_mgmt` |
| `wheat_combined` | `v28_all_mgmt` |
| `sunflower` | `v29_pruned_mgmt` |
| `barley` | `v27_s2` |

## Метрики

| scenario       | model         |   n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:--------------|----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | v30_policy    | 904 | 41.45  |  0.119 |    0.156 |  1.076 | 0.878 |
| all_crops      | cropwise_asof | 901 | 27.829 |  0.44  |    0.18  |  0.857 | 0.666 |
| wheat_combined | v30_policy    | 456 | 33.761 |  0.299 |    0.142 |  0.977 | 0.804 |
| wheat_combined | cropwise_asof | 456 | 26.44  |  0.454 |    0.198 |  0.862 | 0.676 |
| sunflower      | v30_policy    | 209 | 21.817 |  0.217 |    0.182 |  0.67  | 0.508 |
| sunflower      | cropwise_asof | 209 | 23.653 | -0.029 |    0.111 |  0.768 | 0.611 |
| barley         | v30_policy    |  84 | 40.36  | -0.007 |    0.161 |  1.425 | 1.195 |
| barley         | cropwise_asof |  84 | 40.149 |  0.242 |    0.245 |  1.236 | 0.986 |

## Короткий вывод

- `sunflower`: наша policy-модель лучше Cropwise-asof по MAPE и R²; это самый сильный результат.
- `wheat/all_crops`: полный management помогает, но Cropwise-asof всё ещё сильнее.
- `barley`: management шумит; лучшая наша ветка без management почти равна Cropwise по MAPE, но хуже по R².
- Универсальной одной модели нет: лучший результат получается через crop-specific policy.
