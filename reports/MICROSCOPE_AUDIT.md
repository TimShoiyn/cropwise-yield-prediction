# Microscope audit

Goal: validate v17 conclusions before drawing dissertation-level claims.

## 1. Naive baselines vs ML vs Cropwise

Walk-forward OOF on the same rows as the best v17 policy. MAPE is in %, MAE/RMSE in t/ha.

| scenario       | asof_label   |   n_ml |   ml_mape |   cw_mape |   global_mean_mape |   crop_mean_mape |   field_mean_mape |   field_crop_mean_mape |   last_year_mape |
|:---------------|:-------------|-------:|----------:|----------:|-------------------:|-----------------:|------------------:|-----------------------:|-----------------:|
| sunflower      | 1 Jul        |     25 |     25.33 |     31.71 |              24.49 |            24.49 |             25.15 |                  25.15 |            23.86 |
| sunflower      | 1 Aug        |     25 |     25.43 |     26.26 |              24.49 |            24.49 |             25.15 |                  25.15 |            23.86 |
| sunflower      | 1 Sep        |     25 |     26.12 |     18.21 |              24.49 |            24.49 |             25.15 |                  25.15 |            23.86 |
| wheat_combined | 1 Jul        |     55 |     30.17 |     31.36 |              40.6  |            57    |             61.58 |                  62.6  |            55.7  |
| wheat_combined | 1 Aug        |     55 |     28.39 |     28.33 |              40.6  |            57    |             61.58 |                  62.6  |            55.7  |
| wheat_combined | 1 Sep        |     55 |     33.57 |     30.26 |              40.6  |            57    |             61.58 |                  62.6  |            55.7  |
| all_crops      | 1 Jul        |    127 |     28.41 |     30.39 |              34.35 |            39.21 |             36.96 |                  44.77 |            47.16 |
| all_crops      | 1 Aug        |    127 |     29.02 |     30.15 |              34.35 |            39.21 |             36.96 |                  44.77 |            47.16 |
| all_crops      | 1 Sep        |    127 |     30.95 |     28.83 |              34.35 |            39.21 |             36.96 |                  44.77 |            47.16 |

### MAE block (lower is better)

| scenario       | asof_label   |   ml_mae |   cw_mae |   global_mean_mae |   crop_mean_mae |   field_mean_mae |   field_crop_mean_mae |   last_year_mae |
|:---------------|:-------------|---------:|---------:|------------------:|----------------:|-----------------:|----------------------:|----------------:|
| sunflower      | 1 Jul        |    0.602 |    0.697 |             0.577 |           0.577 |            0.571 |                 0.571 |           0.537 |
| sunflower      | 1 Aug        |    0.602 |    0.573 |             0.577 |           0.577 |            0.571 |                 0.571 |           0.537 |
| sunflower      | 1 Sep        |    0.594 |    0.409 |             0.577 |           0.577 |            0.571 |                 0.571 |           0.537 |
| wheat_combined | 1 Jul        |    0.88  |    0.646 |             1.18  |           1.572 |            1.631 |                 1.751 |           1.469 |
| wheat_combined | 1 Aug        |    0.876 |    0.592 |             1.18  |           1.572 |            1.631 |                 1.751 |           1.469 |
| wheat_combined | 1 Sep        |    1.013 |    0.635 |             1.18  |           1.572 |            1.631 |                 1.751 |           1.469 |
| all_crops      | 1 Jul        |    0.78  |    0.669 |             0.968 |           1.047 |            0.994 |                 1.188 |           1.186 |
| all_crops      | 1 Aug        |    0.827 |    0.636 |             0.968 |           1.047 |            0.994 |                 1.188 |           1.186 |
| all_crops      | 1 Sep        |    0.853 |    0.612 |             0.968 |           1.047 |            0.994 |                 1.188 |           1.186 |

### R^2 block

| scenario       | asof_label   |   ml_r2 |   cw_r2 |   global_mean_r2 |   crop_mean_r2 |   field_mean_r2 |   field_crop_mean_r2 |   last_year_r2 |
|:---------------|:-------------|--------:|--------:|-----------------:|---------------:|----------------:|---------------------:|---------------:|
| sunflower      | 1 Jul        |  -0.374 |  -1.111 |           -0.261 |         -0.261 |          -0.524 |               -0.524 |         -0.583 |
| sunflower      | 1 Aug        |  -0.358 |  -0.487 |           -0.261 |         -0.261 |          -0.524 |               -0.524 |         -0.583 |
| sunflower      | 1 Sep        |  -0.384 |   0.264 |           -0.261 |         -0.261 |          -0.524 |               -0.524 |         -0.583 |
| wheat_combined | 1 Jul        |  -0.016 |   0.479 |           -0.622 |         -1.465 |          -1.911 |               -2.278 |         -1.452 |
| wheat_combined | 1 Aug        |  -0.073 |   0.57  |           -0.622 |         -1.465 |          -1.911 |               -2.278 |         -1.452 |
| wheat_combined | 1 Sep        |  -0.272 |   0.521 |           -0.622 |         -1.465 |          -1.911 |               -2.278 |         -1.452 |
| all_crops      | 1 Jul        |  -0.083 |   0.247 |           -0.554 |         -0.879 |          -0.672 |               -1.552 |         -1.439 |
| all_crops      | 1 Aug        |  -0.222 |   0.301 |           -0.554 |         -0.879 |          -0.672 |               -1.552 |         -1.439 |
| all_crops      | 1 Sep        |  -0.242 |   0.341 |           -0.554 |         -0.879 |          -0.672 |               -1.552 |         -1.439 |

## 2. Cropwise as-of timing audit

We pick the latest Cropwise estimate with date <= as-of. If `min_offset_days` is negative, we would be using a future estimate (leakage); if it's always >= 0 we are safe. `median_offset_days` shows how stale the estimate is.

|   asof_tag | asof_label   |   n_field_year_with_history |   n_with_value_at_asof |   share_with_value |   median_offset_days |   p90_offset_days |   max_offset_days |   min_offset_days |
|-----------:|:-------------|----------------------------:|-----------------------:|-------------------:|---------------------:|------------------:|------------------:|------------------:|
|      07_01 | 1 Jul        |                        4759 |                   4759 |                  1 |                    3 |                 6 |               356 |                 0 |
|      08_01 | 1 Aug        |                        4759 |                   4759 |                  1 |                    5 |                17 |               387 |                 0 |
|      09_01 | 1 Sep        |                        4759 |                   4759 |                  1 |                    5 |                48 |               418 |                 0 |

## 3. Walk-forward sample / field distribution per test year

For each scenario / as-of date / test year: number of train rows, train years, test rows, target stats.

| scenario       |   asof_tag |   test_year |   n_train |   n_train_years |   n_test |   test_target_mean |   train_target_mean |   test_target_std |   train_target_std |
|:---------------|-----------:|------------:|----------:|----------------:|---------:|-------------------:|--------------------:|------------------:|-------------------:|
| sunflower      |      08_01 |        2018 |         0 |               0 |        3 |              2.616 |             nan     |             0.878 |            nan     |
| sunflower      |      08_01 |        2019 |         3 |               1 |       14 |              2.484 |               2.616 |             0.735 |              0.878 |
| sunflower      |      08_01 |        2021 |        17 |               2 |       17 |              2.412 |               2.507 |             0.529 |              0.734 |
| sunflower      |      08_01 |        2022 |        34 |               3 |        3 |              1.914 |               2.46  |             1.188 |              0.632 |
| sunflower      |      08_01 |        2023 |        37 |               4 |        4 |              2.598 |               2.415 |             0.215 |              0.684 |
| sunflower      |      08_01 |        2024 |        41 |               5 |       10 |              2.922 |               2.433 |             0.464 |              0.653 |
| sunflower      |      08_01 |        2025 |        51 |               6 |        8 |              2.997 |               2.529 |             0.29  |              0.647 |
| wheat_combined |      08_01 |        2019 |         0 |               0 |        4 |              3.948 |             nan     |             0.537 |            nan     |
| wheat_combined |      08_01 |        2020 |         4 |               1 |       19 |              0.816 |               3.948 |             0.808 |              0.537 |
| wheat_combined |      08_01 |        2021 |        23 |               2 |       10 |              2.901 |               1.361 |             1.17  |              1.43  |
| wheat_combined |      08_01 |        2022 |        33 |               3 |       23 |              2.722 |               1.827 |             0.775 |              1.519 |
| wheat_combined |      08_01 |        2023 |        56 |               4 |       10 |              1.712 |               2.195 |             0.865 |              1.335 |
| wheat_combined |      08_01 |        2024 |        66 |               5 |       12 |              4.192 |               2.122 |             0.841 |              1.281 |
| wheat_combined |      08_01 |        2025 |        78 |               6 |       10 |              3.221 |               2.44  |             0.732 |              1.432 |
| all_crops      |      08_01 |        2018 |         0 |               0 |        8 |              2.16  |             nan     |             1.154 |            nan     |
| all_crops      |      08_01 |        2019 |         8 |               1 |       21 |              2.603 |               2.16  |             1     |              1.154 |
| all_crops      |      08_01 |        2020 |        29 |               2 |       24 |              0.72  |               2.481 |             0.743 |              1.043 |
| all_crops      |      08_01 |        2021 |        53 |               3 |       30 |              2.603 |               1.683 |             0.82  |              1.27  |
| all_crops      |      08_01 |        2022 |        83 |               4 |       26 |              2.629 |               2.016 |             0.843 |              1.208 |
| all_crops      |      08_01 |        2023 |       109 |               5 |       23 |              2.275 |               2.162 |             0.915 |              1.158 |
| all_crops      |      08_01 |        2024 |       132 |               6 |       23 |              3.588 |               2.182 |             0.926 |              1.117 |
| all_crops      |      08_01 |        2025 |       155 |               7 |       25 |              2.878 |               2.39  |             0.882 |              1.198 |

## 4. Feature importance (CatBoost in-sample, 1 Aug)

Top 15 features per scenario. If field_id dominates, the model relies on memorization.

### sunflower

| feature                   |   importance |
|:--------------------------|-------------:|
| ndvi_anom_last_value_asof |         9.55 |
| ndvi_anom_max_asof        |         4.41 |
| wx_precip_sum_to_asof     |         3.32 |
| ndvi_anom_min_asof        |         2.82 |
| ndvi_integral_asof        |         2.82 |
| ndvi_above_05_days_asof   |         2.52 |
| wx_jun_et0                |         2.49 |
| field_soil_P              |         2.22 |
| wx_jul_et0                |         1.99 |
| rotation_pair             |         1.9  |
| ndvi_at_sow_gdd_400       |         1.9  |
| field_soil_K              |         1.87 |
| ndvi_p25_asof             |         1.73 |
| ndvi_max_asof             |         1.69 |
| prev_crop_id              |         1.5  |

### wheat_combined

| feature                  |   importance |
|:-------------------------|-------------:|
| ndvi_p75_asof            |         9.1  |
| ndvi_std_asof            |         8.06 |
| ndvi_anom_mean_asof      |         5.7  |
| wx_precip_sum_to_asof    |         5.36 |
| wx_water_balance_to_asof |         4.97 |
| ndvi_max_asof            |         4.53 |
| ndvi_above_05_days_asof  |         4.46 |
| wx_jul_vpd_max           |         3.63 |
| wx_jun_precip            |         3.52 |
| ndvi_mean_asof           |         3.4  |
| wx_may_srad              |         3.02 |
| ndvi_amplitude_asof      |         2.58 |
| wx_may_et0               |         2.49 |
| ndvi_peak_doy_asof       |         2.36 |
| rotation_pair            |         2.29 |

### all_crops

| feature                     |   importance |
|:----------------------------|-------------:|
| ndvi_std_asof               |         5.27 |
| ndvi_anom_mean_asof         |         4.23 |
| ndvi_integral_asof          |         3.91 |
| wx_water_balance_to_asof    |         3.72 |
| ndvi_p75_asof               |         3.7  |
| wx_may_srad                 |         3.51 |
| ndvi_above_05_days_asof     |         3.48 |
| ndvi_amplitude_asof         |         3.04 |
| crop_id                     |         2.96 |
| ndvi_anom_min_asof          |         2.87 |
| ndvi_anom_max_asof          |         2.81 |
| ndvi_mean_asof              |         2.39 |
| mgmt_chemical_fact_rate_sum |         2.22 |
| ndvi_last_value_asof        |         2.1  |
| wx_vpd_mean_to_asof         |         1.98 |

## 5. In-sample vs OOF gap (over/underfit diagnostic)

If in-sample R^2 is near 1 and OOF R^2 is near 0, the model overfits. If both are near 0, the model has too few signal in features for this dataset size.

| scenario       |   asof_tag |   n_total |   n_oof |   in_sample_mape |   in_sample_r2 |   in_sample_mae |   oof_mape |   oof_r2 |   oof_mae |   mape_gap |   r2_gap |
|:---------------|-----------:|----------:|--------:|-----------------:|---------------:|----------------:|-----------:|---------:|----------:|-----------:|---------:|
| sunflower      |      07_01 |        59 |      25 |            4.69  |          0.947 |           0.109 |     25.333 |   -0.374 |     0.602 |     20.643 |    1.321 |
| sunflower      |      08_01 |        59 |      25 |            1.75  |          0.987 |           0.041 |     25.434 |   -0.358 |     0.602 |     23.683 |    1.346 |
| sunflower      |      09_01 |        59 |      25 |            2.435 |          0.978 |           0.058 |     26.115 |   -0.384 |     0.594 |     23.68  |    1.362 |
| wheat_combined |      07_01 |        88 |      55 |           10.763 |          0.978 |           0.152 |     30.171 |   -0.016 |     0.88  |     19.407 |    0.994 |
| wheat_combined |      08_01 |        88 |      55 |           14.417 |          0.952 |           0.22  |     28.394 |   -0.073 |     0.876 |     13.976 |    1.025 |
| wheat_combined |      09_01 |        88 |      55 |            2.435 |          0.999 |           0.033 |     33.573 |   -0.272 |     1.013 |     31.138 |    1.271 |
| all_crops      |      07_01 |       180 |     127 |           21.033 |          0.834 |           0.374 |     28.412 |   -0.083 |     0.78  |      7.38  |    0.917 |
| all_crops      |      08_01 |       180 |     127 |           12.485 |          0.948 |           0.209 |     29.018 |   -0.222 |     0.827 |     16.533 |    1.17  |
| all_crops      |      09_01 |       180 |     127 |           13.382 |          0.945 |           0.221 |     30.951 |   -0.242 |     0.853 |     17.568 |    1.187 |

## 6. Per-year breakdown (target year, 1 Aug only for brevity)

| scenario       |   asof_tag |   year |   n |   n_ml |   n_cw |   target_mean |   ml_mape |   ml_mae |   cw_mape |   cw_mae |
|:---------------|-----------:|-------:|----:|-------:|-------:|--------------:|----------:|---------:|----------:|---------:|
| all_crops      |      08_01 |   2018 |   8 |      0 |      8 |         2.16  |   nan     |  nan     |    41.206 |    0.704 |
| all_crops      |      08_01 |   2019 |  21 |      0 |     21 |         2.603 |   nan     |  nan     |    20.953 |    0.523 |
| all_crops      |      08_01 |   2020 |  24 |      0 |     24 |         0.72  |   nan     |  nan     |   322.441 |    1.227 |
| all_crops      |      08_01 |   2021 |  30 |     30 |     30 |         2.603 |    25.861 |    0.77  |    31.781 |    0.681 |
| all_crops      |      08_01 |   2022 |  26 |     26 |     26 |         2.629 |    32.093 |    0.847 |    30.15  |    0.592 |
| all_crops      |      08_01 |   2023 |  23 |     23 |     23 |         2.275 |    35.042 |    0.927 |    41.03  |    0.732 |
| all_crops      |      08_01 |   2024 |  23 |     23 |     23 |         3.588 |    22.322 |    0.895 |    13.712 |    0.435 |
| all_crops      |      08_01 |   2025 |  25 |     25 |     25 |         2.878 |    30.228 |    0.72  |    33.317 |    0.724 |
| sunflower      |      08_01 |   2018 |   3 |      0 |      3 |         2.616 |   nan     |  nan     |    33.47  |    0.849 |
| sunflower      |      08_01 |   2019 |  14 |      0 |     14 |         2.484 |   nan     |  nan     |    21.86  |    0.569 |
| sunflower      |      08_01 |   2021 |  17 |      0 |     17 |         2.412 |   nan     |  nan     |    22.198 |    0.474 |
| sunflower      |      08_01 |   2022 |   3 |      3 |      3 |         1.914 |    79.521 |    1.081 |    87.624 |    1.18  |
| sunflower      |      08_01 |   2023 |   4 |      4 |      4 |         2.598 |     9.138 |    0.243 |    20.813 |    0.543 |
| sunflower      |      08_01 |   2024 |  10 |     10 |     10 |         2.922 |    19.921 |    0.589 |    14.787 |    0.392 |
| sunflower      |      08_01 |   2025 |   8 |      8 |      8 |         2.997 |    20.19  |    0.618 |    20.305 |    0.588 |
| wheat_combined |      08_01 |   2019 |   4 |      0 |      4 |         3.948 |   nan     |  nan     |    13.973 |    0.519 |
| wheat_combined |      08_01 |   2020 |  19 |      0 |     19 |         0.816 |   nan     |  nan     |   288.666 |    1.191 |
| wheat_combined |      08_01 |   2021 |  10 |      0 |     10 |         2.901 |   nan     |  nan     |    45.911 |    0.957 |
| wheat_combined |      08_01 |   2022 |  23 |     23 |     23 |         2.722 |    33.643 |    1.03  |    22.653 |    0.516 |
| wheat_combined |      08_01 |   2023 |  10 |     10 |     10 |         1.712 |    30.793 |    0.489 |    69.081 |    0.935 |
| wheat_combined |      08_01 |   2024 |  12 |     12 |     12 |         4.192 |    22.976 |    1.115 |    11.996 |    0.449 |
| wheat_combined |      08_01 |   2025 |  10 |     10 |     10 |         3.221 |    20.422 |    0.621 |    20.22  |    0.596 |

## 8. physical_t_ha sanity (harvested_weight / tillable_area)

- rows with both fields populated: 131
- mean: 2.784
- median: 2.859
- min: 0.890
- max: 5.519
- abnormally low (<0.3 t/ha): 0
- abnormally high (>12 t/ha): 0

### Top 15 abnormally low (potential partial harvest)

| field_id   | year   | standard_name   | harvested_weight   | tillable_area   | physical_t_ha   | productivity   |
|------------|--------|-----------------|--------------------|-----------------|-----------------|----------------|

### Top 15 abnormally high

| field_id   | year   | standard_name   | harvested_weight   | tillable_area   | physical_t_ha   | productivity   |
|------------|--------|-----------------|--------------------|-----------------|-----------------|----------------|

