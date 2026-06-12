# v28 — management feature ablation

Дата: 2026-06-11.

Что добавили: as-of safe management-фичи из `agro_operations`:

- counts операций до даты прогноза (`application`, `soil`, `other`);
- дни с последней операции/обработки;
- нормы удобрений и оценка N/P2O5/K2O/S кг/га через `fertilizers`;
- нормы химии: herbicide/fungicide/insecticide через `chemicals`;
- seed rate.

Важно: `planned/canceled` исключены, harvest исключён, берём только `actual_start_datetime <= asof`.
То есть лика уборки нет.

## Короткий вывод

Management-фичи **не дают универсального прорыва**, но дают полезный сигнал на главной дате **1 августа**, особенно для wheat/all_crops:

- `all_crops 1 Aug`: MAPE **47.0% → 41.5%**, R² **-0.05 → 0.12**;
- `wheat 1 Aug`: MAPE **38.5% → 33.8%**, R² **0.19 → 0.30**;
- `sunflower`: почти без изменений;
- `barley`: ухудшение, вероятно из-за малой выборки и шумных/плановых норм.

Cropwise всё ещё сильнее на wheat/all_crops, но на sunflower наша модель остаётся лучше Cropwise-asof по MAPE.

## Полная таблица

Fair ablation on identical S2-covered, pre-harvest rows. No field_id.

| scenario       | asof   | model         |   n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:-------|:--------------|----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | 1 Jul  | v27_s2        | 946 | 42.291 |  0.155 |    0.141 |  1.095 | 0.901 |
| all_crops      | 1 Jul  | v28_mgmt      | 946 | 41.895 |  0.15  |    0.146 |  1.098 | 0.898 |
| all_crops      | 1 Jul  | cropwise_asof | 943 | 29.348 |  0.333 |    0.025 |  0.972 | 0.751 |
| all_crops      | 1 Aug  | v27_s2        | 904 | 46.976 | -0.051 |    0.04  |  1.175 | 0.966 |
| all_crops      | 1 Aug  | v28_mgmt      | 904 | 41.45  |  0.119 |    0.156 |  1.076 | 0.878 |
| all_crops      | 1 Aug  | cropwise_asof | 901 | 27.829 |  0.44  |    0.18  |  0.857 | 0.666 |
| all_crops      | 1 Sep  | v27_s2        |  63 | 22.123 |  0.146 |    0.149 |  0.762 | 0.611 |
| all_crops      | 1 Sep  | v28_mgmt      |  63 | 21.939 |  0.168 |    0.171 |  0.752 | 0.592 |
| all_crops      | 1 Sep  | cropwise_asof |  63 | 24.713 |  0.083 |    0.068 |  0.79  | 0.629 |
| wheat_combined | 1 Jul  | v27_s2        | 490 | 33.801 |  0.308 |    0.124 |  1.035 | 0.835 |
| wheat_combined | 1 Jul  | v28_mgmt      | 490 | 37.168 |  0.211 |    0.108 |  1.104 | 0.906 |
| wheat_combined | 1 Jul  | cropwise_asof | 490 | 28.367 |  0.338 |    0.014 |  1.012 | 0.772 |
| wheat_combined | 1 Aug  | v27_s2        | 456 | 38.495 |  0.194 |    0.153 |  1.047 | 0.877 |
| wheat_combined | 1 Aug  | v28_mgmt      | 456 | 33.761 |  0.299 |    0.142 |  0.977 | 0.804 |
| wheat_combined | 1 Aug  | cropwise_asof | 456 | 26.44  |  0.454 |    0.198 |  0.862 | 0.676 |
| sunflower      | 1 Jul  | v27_s2        | 210 | 24.071 |  0.033 |    0.077 |  0.743 | 0.572 |
| sunflower      | 1 Jul  | v28_mgmt      | 210 | 23.972 |  0.075 |    0.106 |  0.727 | 0.557 |
| sunflower      | 1 Jul  | cropwise_asof | 210 | 25.657 | -0.335 |    0.003 |  0.873 | 0.71  |
| sunflower      | 1 Aug  | v27_s2        | 209 | 22.604 |  0.179 |    0.147 |  0.686 | 0.526 |
| sunflower      | 1 Aug  | v28_mgmt      | 209 | 22.606 |  0.18  |    0.141 |  0.686 | 0.526 |
| sunflower      | 1 Aug  | cropwise_asof | 209 | 23.653 | -0.029 |    0.111 |  0.768 | 0.611 |
| barley         | 1 Jul  | v27_s2        |  83 | 39.374 |  0.083 |    0.289 |  1.355 | 1.149 |
| barley         | 1 Jul  | v28_mgmt      |  83 | 41.162 |  0.003 |    0.252 |  1.413 | 1.18  |
| barley         | 1 Jul  | cropwise_asof |  83 | 42.438 |  0.106 |    0.116 |  1.337 | 1.109 |
| barley         | 1 Aug  | v27_s2        |  84 | 40.36  | -0.007 |    0.161 |  1.425 | 1.195 |
| barley         | 1 Aug  | v28_mgmt      |  84 | 43.552 | -0.231 |    0.171 |  1.575 | 1.342 |
| barley         | 1 Aug  | cropwise_asof |  84 | 40.149 |  0.242 |    0.245 |  1.236 | 0.986 |

## Что делать дальше

Не надо тащить все management-фичи слепо в финальную модель. Разумнее:

1. оставить их для wheat/all_crops 1 Aug;
2. для sunflower management почти не нужен;
3. для barley management выключить или сильно prune;
4. сделать v29-pruned: только устойчивые management-фичи (`application_count`, `days_since_last_application`, `N/P/S`, `herbicide/fungicide/insecticide rates`), без шумных totals.
