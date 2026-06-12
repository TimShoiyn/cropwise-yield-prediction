# Audit 2: R2 instability decomposition

Pooled R2 mixes between-crop level, between-year level, and within crop-year field ranking. The honest local skill is `within_crop_year_r2`.

## R2 decomposition (peer transfer ML)

| scenario       | asof   |   n |   pooled_r2 |   pooled_r2_no2020 |   within_year_r2 |   within_crop_r2 |   within_crop_year_r2 |
|:---------------|:-------|----:|------------:|-------------------:|-----------------:|-----------------:|----------------------:|
| sunflower      | 1 Jul  |  59 |      -0.069 |             -0.069 |           -0.025 |           -0.053 |                -0.025 |
| sunflower      | 1 Aug  |  59 |      -0.411 |             -0.411 |            0.043 |           -0.042 |                 0.043 |
| sunflower      | 1 Sep  |  59 |      -1.081 |             -1.081 |           -0.021 |           -0.104 |                -0.021 |
| wheat_combined | 1 Jul  |  88 |       0.212 |              0.176 |            0.087 |            0.464 |                 0.049 |
| wheat_combined | 1 Aug  |  88 |       0.303 |              0.373 |            0.167 |            0.511 |                 0.084 |
| wheat_combined | 1 Sep  |  88 |       0.507 |              0.479 |            0.206 |            0.546 |                 0.055 |
| all_crops      | 1 Jul  | 180 |       0.095 |              0.181 |            0.193 |            0.426 |                 0.083 |
| all_crops      | 1 Aug  | 180 |       0.272 |              0.259 |            0.11  |            0.402 |                 0.011 |
| all_crops      | 1 Sep  | 180 |       0.286 |              0.22  |            0.047 |            0.426 |                -0.007 |

## R2 decomposition (Cropwise)

| scenario       | asof   |   n |   pooled_r2 |   pooled_r2_no2020 |   within_year_r2 |   within_crop_r2 |   within_crop_year_r2 |
|:---------------|:-------|----:|------------:|-------------------:|-----------------:|-----------------:|----------------------:|
| sunflower      | 1 Jul  |  59 |      -0.477 |             -0.477 |           -0.338 |           -0.306 |                -0.338 |
| sunflower      | 1 Aug  |  59 |      -0.174 |             -0.174 |           -0.242 |           -0.167 |                -0.242 |
| sunflower      | 1 Sep  |  59 |      -0     |             -0     |           -0.084 |            0.007 |                -0.084 |
| wheat_combined | 1 Jul  |  88 |       0.5   |              0.445 |            0.141 |            0.552 |                -0.087 |
| wheat_combined | 1 Aug  |  88 |       0.544 |              0.481 |            0.188 |            0.639 |                 0.035 |
| wheat_combined | 1 Sep  |  88 |       0.539 |              0.464 |            0.157 |            0.624 |                -0.059 |
| all_crops      | 1 Jul  | 180 |       0.11  |             -0.072 |           -0.153 |            0.405 |                -0.124 |
| all_crops      | 1 Aug  | 180 |       0.441 |              0.365 |            0.28  |            0.483 |                 0.006 |
| all_crops      | 1 Sep  | 180 |       0.432 |              0.352 |            0.277 |            0.486 |                -0.071 |

## Per-year R2 (2020 leverage), 1 Aug

| scenario       | asof   |   year |   n |   target_mean |   target_std |   ml_r2 |   cw_r2 |
|:---------------|:-------|-------:|----:|--------------:|-------------:|--------:|--------:|
| sunflower      | 1 Aug  |   2018 |   3 |         2.616 |        0.878 |  -0.132 |  -0.527 |
| sunflower      | 1 Aug  |   2019 |  14 |         2.484 |        0.735 |  -0.091 |   0.06  |
| sunflower      | 1 Aug  |   2021 |  17 |         2.412 |        0.529 |   0.081 |  -0.229 |
| sunflower      | 1 Aug  |   2022 |   3 |         1.914 |        1.188 |   0.074 |  -0.613 |
| sunflower      | 1 Aug  |   2023 |   4 |         2.598 |        0.215 | -11.102 |  -8.675 |
| sunflower      | 1 Aug  |   2024 |  10 |         2.922 |        0.464 |  -2.746 |  -0.428 |
| sunflower      | 1 Aug  |   2025 |   8 |         2.997 |        0.29  | -11.597 |  -5.621 |
| wheat_combined | 1 Aug  |   2019 |   4 |         3.948 |        0.537 |  -0.071 |  -0.642 |
| wheat_combined | 1 Aug  |   2020 |  19 |         0.816 |        0.808 |  -4.428 |  -1.754 |
| wheat_combined | 1 Aug  |   2021 |  10 |         2.901 |        1.17  |  -0.144 |  -0.096 |
| wheat_combined | 1 Aug  |   2022 |  23 |         2.722 |        0.775 |  -0.346 |   0.229 |
| wheat_combined | 1 Aug  |   2023 |  10 |         1.712 |        0.865 |  -0.691 |  -0.529 |
| wheat_combined | 1 Aug  |   2024 |  12 |         4.192 |        0.841 |   0.396 |   0.543 |
| wheat_combined | 1 Aug  |   2025 |  10 |         3.221 |        0.732 |   0.052 |  -0.1   |
| all_crops      | 1 Aug  |   2018 |   8 |         2.16  |        1.154 |  -0.938 |   0.502 |
| all_crops      | 1 Aug  |   2019 |  21 |         2.603 |        1     |   0.576 |   0.568 |
| all_crops      | 1 Aug  |   2020 |  24 |         0.72  |        0.743 |  -4.374 |  -2.358 |
| all_crops      | 1 Aug  |   2021 |  30 |         2.603 |        0.82  |  -0.125 |  -0.121 |
| all_crops      | 1 Aug  |   2022 |  26 |         2.629 |        0.843 |   0.123 |   0.171 |
| all_crops      | 1 Aug  |   2023 |  23 |         2.275 |        0.915 |   0.047 |   0.05  |
| all_crops      | 1 Aug  |   2024 |  23 |         3.588 |        0.926 |   0.497 |   0.638 |
| all_crops      | 1 Aug  |   2025 |  25 |         2.878 |        0.882 |  -0.012 |  -0.078 |

## Reading guide

- If `pooled_r2` >> `within_crop_year_r2`, R2 mostly explains crop/year level, not field ranking.
- If `pooled_r2_no2020` differs a lot from `pooled_r2`, 2020 is a leverage point.
- If `within_crop_year_r2` ~ 0 for everyone incl. Cropwise, nobody ranks fields inside a crop-year here.
