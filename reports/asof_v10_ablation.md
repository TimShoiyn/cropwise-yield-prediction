# v10 Internal Feature Ablation

Same model params (`mae_l30`), varying feature groups.

## Best by scenario/date

| scenario       |   asof_tag | asof_label   | variant      |   n |     r2 |   rmse |   mae |   mape |
|:---------------|-----------:|:-------------|:-------------|----:|-------:|-------:|------:|-------:|
| all_crops      |      07_01 | 1 Jul        | v9           | 155 | -0.305 |  1.257 | 0.943 | 34.613 |
| all_crops      |      08_01 | 1 Aug        | v9           | 155 | -0.144 |  1.177 | 0.873 | 32.55  |
| all_crops      |      09_01 | 1 Sep        | v10_no_soil  | 155 | -0.215 |  1.213 | 0.902 | 33.975 |
| sunflower      |      07_01 | 1 Jul        | v10_full     |  25 | -0.438 |  0.697 | 0.592 | 24.348 |
| sunflower      |      08_01 | 1 Aug        | v10_no_scout |  25 | -0.306 |  0.665 | 0.578 | 23.534 |
| sunflower      |      09_01 | 1 Sep        | v10_no_scout |  25 | -0.287 |  0.66  | 0.582 | 24.77  |
| wheat_combined |      07_01 | 1 Jul        | v9           |  67 | -0.001 |  1.131 | 0.893 | 37.347 |
| wheat_combined |      08_01 | 1 Aug        | v10_no_soil  |  67 |  0.027 |  1.115 | 0.905 | 38.971 |
| wheat_combined |      09_01 | 1 Sep        | v9           |  67 |  0.119 |  1.061 | 0.853 | 38.993 |

## Full ablation grid

| scenario       |   asof_tag | asof_label   | variant         |   n |     r2 |   rmse |   mae |   mape |
|:---------------|-----------:|:-------------|:----------------|----:|-------:|-------:|------:|-------:|
| sunflower      |      07_01 | 1 Jul        | v9              |  25 | -0.647 |  0.746 | 0.651 | 26.287 |
| sunflower      |      07_01 | 1 Jul        | v10_full        |  25 | -0.438 |  0.697 | 0.592 | 24.348 |
| sunflower      |      07_01 | 1 Jul        | v10_no_scout    |  25 | -0.575 |  0.73  | 0.641 | 25.876 |
| sunflower      |      07_01 | 1 Jul        | v10_no_soil     |  25 | -0.896 |  0.801 | 0.695 | 27.16  |
| sunflower      |      07_01 | 1 Jul        | v10_no_internal |  25 | -0.647 |  0.746 | 0.651 | 26.287 |
| sunflower      |      08_01 | 1 Aug        | v9              |  25 | -0.581 |  0.731 | 0.652 | 25.86  |
| sunflower      |      08_01 | 1 Aug        | v10_full        |  25 | -0.429 |  0.695 | 0.62  | 24.811 |
| sunflower      |      08_01 | 1 Aug        | v10_no_scout    |  25 | -0.306 |  0.665 | 0.578 | 23.534 |
| sunflower      |      08_01 | 1 Aug        | v10_no_soil     |  25 | -0.346 |  0.675 | 0.578 | 24.933 |
| sunflower      |      08_01 | 1 Aug        | v10_no_internal |  25 | -0.581 |  0.731 | 0.652 | 25.86  |
| sunflower      |      09_01 | 1 Sep        | v9              |  25 | -0.328 |  0.67  | 0.578 | 24.9   |
| sunflower      |      09_01 | 1 Sep        | v10_full        |  25 | -0.397 |  0.687 | 0.602 | 25.601 |
| sunflower      |      09_01 | 1 Sep        | v10_no_scout    |  25 | -0.287 |  0.66  | 0.582 | 24.77  |
| sunflower      |      09_01 | 1 Sep        | v10_no_soil     |  25 | -0.448 |  0.7   | 0.612 | 26.7   |
| sunflower      |      09_01 | 1 Sep        | v10_no_internal |  25 | -0.328 |  0.67  | 0.578 | 24.9   |
| wheat_combined |      07_01 | 1 Jul        | v9              |  67 | -0.001 |  1.131 | 0.893 | 37.347 |
| wheat_combined |      07_01 | 1 Jul        | v10_full        |  67 | -0.12  |  1.197 | 0.955 | 38.805 |
| wheat_combined |      07_01 | 1 Jul        | v10_no_scout    |  67 |  0.006 |  1.127 | 0.914 | 38.387 |
| wheat_combined |      07_01 | 1 Jul        | v10_no_soil     |  67 |  0.008 |  1.126 | 0.913 | 39.865 |
| wheat_combined |      07_01 | 1 Jul        | v10_no_internal |  67 | -0.001 |  1.131 | 0.893 | 37.347 |
| wheat_combined |      08_01 | 1 Aug        | v9              |  67 |  0.076 |  1.087 | 0.883 | 39.132 |
| wheat_combined |      08_01 | 1 Aug        | v10_full        |  67 | -0.036 |  1.151 | 0.933 | 41.736 |
| wheat_combined |      08_01 | 1 Aug        | v10_no_scout    |  67 | -0.009 |  1.136 | 0.919 | 40.79  |
| wheat_combined |      08_01 | 1 Aug        | v10_no_soil     |  67 |  0.027 |  1.115 | 0.905 | 38.971 |
| wheat_combined |      08_01 | 1 Aug        | v10_no_internal |  67 |  0.076 |  1.087 | 0.883 | 39.132 |
| wheat_combined |      09_01 | 1 Sep        | v9              |  67 |  0.119 |  1.061 | 0.853 | 38.993 |
| wheat_combined |      09_01 | 1 Sep        | v10_full        |  67 |  0.012 |  1.124 | 0.906 | 39.648 |
| wheat_combined |      09_01 | 1 Sep        | v10_no_scout    |  67 | -0.131 |  1.203 | 0.962 | 42.378 |
| wheat_combined |      09_01 | 1 Sep        | v10_no_soil     |  67 | -0.067 |  1.168 | 0.949 | 42.152 |
| wheat_combined |      09_01 | 1 Sep        | v10_no_internal |  67 |  0.119 |  1.061 | 0.853 | 38.993 |
| all_crops      |      07_01 | 1 Jul        | v9              | 155 | -0.305 |  1.257 | 0.943 | 34.613 |
| all_crops      |      07_01 | 1 Jul        | v10_full        | 155 | -0.333 |  1.27  | 0.954 | 36.731 |
| all_crops      |      07_01 | 1 Jul        | v10_no_scout    | 155 | -0.3   |  1.255 | 0.947 | 36.158 |
| all_crops      |      07_01 | 1 Jul        | v10_no_soil     | 155 | -0.269 |  1.24  | 0.929 | 34.842 |
| all_crops      |      07_01 | 1 Jul        | v10_no_internal | 155 | -0.305 |  1.257 | 0.943 | 34.613 |
| all_crops      |      08_01 | 1 Aug        | v9              | 155 | -0.144 |  1.177 | 0.873 | 32.55  |
| all_crops      |      08_01 | 1 Aug        | v10_full        | 155 | -0.349 |  1.278 | 0.977 | 35.643 |
| all_crops      |      08_01 | 1 Aug        | v10_no_scout    | 155 | -0.334 |  1.271 | 0.959 | 36.048 |
| all_crops      |      08_01 | 1 Aug        | v10_no_soil     | 155 | -0.234 |  1.223 | 0.932 | 34.564 |
| all_crops      |      08_01 | 1 Aug        | v10_no_internal | 155 | -0.144 |  1.177 | 0.873 | 32.55  |
| all_crops      |      09_01 | 1 Sep        | v9              | 155 | -0.306 |  1.258 | 0.947 | 35.794 |
| all_crops      |      09_01 | 1 Sep        | v10_full        | 155 | -0.276 |  1.243 | 0.936 | 34.663 |
| all_crops      |      09_01 | 1 Sep        | v10_no_scout    | 155 | -0.329 |  1.269 | 0.962 | 35.953 |
| all_crops      |      09_01 | 1 Sep        | v10_no_soil     | 155 | -0.215 |  1.213 | 0.902 | 33.975 |
| all_crops      |      09_01 | 1 Sep        | v10_no_internal | 155 | -0.306 |  1.258 | 0.947 | 35.794 |
