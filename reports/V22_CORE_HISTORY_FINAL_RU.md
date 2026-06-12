# v22: core local model + historical field baseline

Validation: walk-forward by year. `v22_core` uses 24 compact features; `v22_core_history` adds only past-year field yield baseline features. `wcy_r2` = within-crop-year R2.

| scenario       |   asof_tag | asof   | model            |   n |     r2 |   wcy_r2 |   rmse |   mae |    mape |
|:---------------|-----------:|:-------|:-----------------|----:|-------:|---------:|-------:|------:|--------:|
| all_crops      |      07_01 | 1 Jul  | v22_core         | 151 |  0.102 |   -0.016 |  1.129 | 0.915 |  96.015 |
| all_crops      |      07_01 | 1 Jul  | v22_core_history | 151 | -0.095 |   -0.41  |  1.247 | 0.992 | 103.35  |
| all_crops      |      07_01 | 1 Jul  | cropwise         | 180 |  0.11  |   -0.124 |  1.1   | 0.803 |  78.571 |
| all_crops      |      08_01 | 1 Aug  | v22_core         | 151 |  0.048 |    0.157 |  1.162 | 0.925 |  98.308 |
| all_crops      |      08_01 | 1 Aug  | v22_core_history | 151 | -0.126 |   -0.412 |  1.264 | 1.001 | 104.423 |
| all_crops      |      08_01 | 1 Aug  | cropwise         | 180 |  0.441 |    0.006 |  0.872 | 0.705 |  68.542 |
| all_crops      |      09_01 | 1 Sep  | v22_core         | 151 | -0.196 |   -0.026 |  1.303 | 1.039 | 118.722 |
| all_crops      |      09_01 | 1 Sep  | v22_core_history | 151 | -0.192 |   -0.501 |  1.301 | 1.036 | 105.841 |
| all_crops      |      09_01 | 1 Sep  | cropwise         | 180 |  0.432 |   -0.071 |  0.879 | 0.7   |  67.838 |
| wheat_combined |      07_01 | 1 Jul  | v22_core         |  65 |  0.196 |   -0.074 |  0.999 | 0.764 |  27.418 |
| wheat_combined |      07_01 | 1 Jul  | v22_core_history |  65 | -0.087 |   -0.244 |  1.161 | 0.903 |  30.939 |
| wheat_combined |      07_01 | 1 Jul  | cropwise         |  88 |  0.5   |   -0.087 |  0.978 | 0.785 |  96.228 |
| wheat_combined |      08_01 | 1 Aug  | v22_core         |  65 |  0.232 |    0.055 |  0.976 | 0.759 |  27.84  |
| wheat_combined |      08_01 | 1 Aug  | v22_core_history |  65 | -0.13  |   -0.483 |  1.184 | 0.908 |  31.207 |
| wheat_combined |      08_01 | 1 Aug  | cropwise         |  88 |  0.544 |    0.035 |  0.933 | 0.759 |  85.883 |
| wheat_combined |      09_01 | 1 Sep  | v22_core         |  65 |  0.118 |    0.149 |  1.046 | 0.821 |  29.465 |
| wheat_combined |      09_01 | 1 Sep  | v22_core_history |  65 | -0.38  |   -0.396 |  1.308 | 1.029 |  36.338 |
| wheat_combined |      09_01 | 1 Sep  | cropwise         |  88 |  0.539 |   -0.059 |  0.939 | 0.777 |  86.232 |
| sunflower      |      07_01 | 1 Jul  | v22_core         |  25 | -0.556 |    0.048 |  0.725 | 0.625 |  24.169 |
| sunflower      |      07_01 | 1 Jul  | v22_core_history |  25 | -0.289 |   -0.097 |  0.66  | 0.563 |  23.851 |
| sunflower      |      07_01 | 1 Jul  | cropwise         |  59 | -0.477 |   -0.338 |  0.76  | 0.619 |  26.147 |
| sunflower      |      08_01 | 1 Aug  | v22_core         |  25 | -0.687 |   -0.173 |  0.755 | 0.656 |  26.816 |
| sunflower      |      08_01 | 1 Aug  | v22_core_history |  25 | -0.449 |   -0.388 |  0.7   | 0.547 |  25.01  |
| sunflower      |      08_01 | 1 Aug  | cropwise         |  59 | -0.174 |   -0.242 |  0.677 | 0.557 |  24.411 |
| sunflower      |      09_01 | 1 Sep  | v22_core         |  25 | -0.246 |   -0.156 |  0.649 | 0.533 |  23.601 |
| sunflower      |      09_01 | 1 Sep  | v22_core_history |  25 | -0.483 |   -0.319 |  0.708 | 0.565 |  25.71  |
| sunflower      |      09_01 | 1 Sep  | cropwise         |  59 | -0     |   -0.084 |  0.625 | 0.499 |  21.866 |

## Winners by scenario/date

- all_crops 1 Jul: best MAPE = **cropwise** (78.6%), best wcy_R2 = **v22_core** (-0.016).
- all_crops 1 Aug: best MAPE = **cropwise** (68.5%), best wcy_R2 = **v22_core** (0.157).
- all_crops 1 Sep: best MAPE = **cropwise** (67.8%), best wcy_R2 = **v22_core** (-0.026).
- wheat_combined 1 Jul: best MAPE = **v22_core** (27.4%), best wcy_R2 = **v22_core** (-0.074).
- wheat_combined 1 Aug: best MAPE = **v22_core** (27.8%), best wcy_R2 = **v22_core** (0.055).
- wheat_combined 1 Sep: best MAPE = **v22_core** (29.5%), best wcy_R2 = **v22_core** (0.149).
- sunflower 1 Jul: best MAPE = **v22_core_history** (23.9%), best wcy_R2 = **v22_core** (0.048).
- sunflower 1 Aug: best MAPE = **cropwise** (24.4%), best wcy_R2 = **v22_core** (-0.173).
- sunflower 1 Sep: best MAPE = **cropwise** (21.9%), best wcy_R2 = **cropwise** (-0.084).

## Практический вывод

### 1. Core-фичи — оставить как основную локальную модель

`v22_core` — самый разумный локальный вариант без новых данных:

- wheat: стабильно лучший MAPE против Cropwise на всех датах (27–29% против 86–96% на всех строках; 26–27% против 27–30% на фильтре `target >= 1`);
- all_crops: иногда лучше по `wcy_r2`, но Cropwise лучше по pooled R² и MAPE;
- sunflower: примерно сопоставим по MAPE, но R² остаётся плохим из-за низкого потолка сигнала.

Главное: 24 core-признака лучше, чем 148 sparse-признаков. Это подтверждает аудит-3b:
маленький датасет + много пустых scout/soil фич = переобучение.

### 2. Historical field baseline НЕ включать

`v22_core_history` почти везде ухудшил результат:

- all_crops: R² 0.048 → -0.126 на 1 Aug;
- wheat: R² 0.232 → -0.130 на 1 Aug;
- wheat MAPE: 27.8% → 31.2%;
- `wcy_r2` резко падает.

Это финально подтверждает прошлый вывод: прошлые урожаи поля **не дают устойчивого
сигнала** для текущего года. Field-effect не персистентен, поэтому добавление
истории поля не помогает, а добавляет шум.

### 3. Что можно писать в диссертации

Лучший честный тезис:

> Упрощённая локальная модель на компактном наборе NDVI/weather/sowing признаков
> превосходит Cropwise по MAPE для пшеницы, но уступает по pooled R², так как
> Cropwise лучше улавливает общий уровень года. Добавление исторического baseline
> поля не улучшает качество, что подтверждает отсутствие устойчивого поля-эффекта
> в текущих данных.

### 4. Следующий разумный технический шаг

Не добавлять больше табличных фич. Следующий настоящий прирост возможен только через
новые remote-sensing признаки:

1. Sentinel-2 red-edge / NDRE / NIRv;
2. текстурные признаки внутри полигона;
3. геометрия остальных полей;
4. yield maps / внутриполевая почва.

Без этого для sunflower и within-crop-year ranking потолок уже достигнут.
