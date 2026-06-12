# Audit 3: full pipeline audit (local v17, 1 Aug)

Rows: 180, candidate numeric features: 148

## 1. Target per crop (t/ha)

| standard_name        |   count |   min |   median |   max |   std |
|:---------------------|--------:|------:|---------:|------:|------:|
| barley_spring        |      15 |  1.53 |     2.53 |  4.5  |  0.83 |
| maize                |       3 |  0.58 |     3.31 |  3.31 |  1.58 |
| oil_seed_raps_spring |       2 |  0.44 |     0.8  |  1.16 |  0.51 |
| oil_seed_raps_winter |       4 |  0.17 |     0.29 |  0.98 |  0.37 |
| pea                  |       4 |  1.33 |     1.64 |  1.81 |  0.21 |
| soya                 |       4 |  0.9  |     1.2  |  1.47 |  0.32 |
| sunflower            |      59 |  1    |     2.69 |  3.96 |  0.63 |
| wheat_spring         |      54 |  0.21 |     2.85 |  4.56 |  1.27 |
| wheat_winter         |      34 |  0.21 |     2.32 |  5.52 |  1.57 |

## 2. Missingness

- Features >=80% missing (effectively dead): **7**
- Features 40-80% missing (weak/sparse): **18**

Top 25 most-missing features:

|                                 |   frac_missing |
|:--------------------------------|---------------:|
| soil_sample_latest_soil_B       |          0.861 |
| soil_sample_mean_soil_B         |          0.861 |
| soil_sample_latest_soil_N       |          0.861 |
| soil_sample_mean_soil_N         |          0.861 |
| ndvi_at_das_120                 |          0.856 |
| ndvi_at_sow_gdd_1200            |          0.85  |
| ndvi_at_das_90                  |          0.822 |
| scout_growth_stage_max          |          0.656 |
| scout_growth_stage_last         |          0.656 |
| scout_condition_score_last      |          0.594 |
| scout_condition_score_min       |          0.594 |
| scout_last_report_doy           |          0.528 |
| scout_days_since_last_report    |          0.528 |
| scout_threat_weed_any           |          0.528 |
| scout_risk_yield_decreasing_n   |          0.528 |
| scout_threat_reports_n          |          0.528 |
| scout_threat_disease_any        |          0.528 |
| scout_risk_yield_decreasing_any |          0.528 |
| scout_threat_total_n            |          0.528 |
| scout_condition_not_good_n      |          0.528 |
| scout_condition_bad_n           |          0.528 |
| scout_threat_insect_any         |          0.528 |
| ndvi_at_das_75                  |          0.439 |
| soil_sample_latest_soil_N_NO3   |          0.406 |
| ndvi_at_sow_gdd_1000            |          0.406 |

## 3. Top feature signal (|Spearman| vs target)

### all

| feature                   |   spearman |   n |
|:--------------------------|-----------:|----:|
| ndvi_at_das_120           |      0.692 |  26 |
| ndvi_anom_mean_asof       |      0.558 | 180 |
| ndvi_p75_asof             |      0.551 | 180 |
| ndvi_max_asof             |      0.54  | 180 |
| ndvi_integral_asof        |      0.527 | 180 |
| ndvi_above_05_days_asof   |      0.518 | 180 |
| ndvi_sow_peak_gdd         |     -0.497 | 130 |
| ndvi_at_sow_gdd_600       |      0.493 | 130 |
| wx_may_precip             |      0.487 | 180 |
| scout_condition_bad_n     |     -0.486 |  85 |
| ndvi_amplitude_asof       |      0.48  | 180 |
| scout_condition_score_min |      0.469 |  73 |

### wheat

| feature                  |   spearman |   n |
|:-------------------------|-----------:|----:|
| ndvi_p75_asof            |      0.813 |  88 |
| ndvi_anom_mean_asof      |      0.771 |  88 |
| ndvi_max_asof            |      0.77  |  88 |
| wx_may_srad              |     -0.757 |  88 |
| wx_jul_vpd_max           |     -0.721 |  88 |
| wx_may_precip            |      0.719 |  88 |
| ndvi_integral_asof       |      0.716 |  88 |
| wx_water_balance_to_asof |      0.71  |  88 |
| ndvi_at_sow_gdd_600      |      0.703 |  66 |
| ndvi_at_das_120          |      0.692 |  26 |
| ndvi_at_das_90           |      0.684 |  26 |
| ndvi_amplitude_asof      |      0.683 |  88 |

### sunflower

| feature                              |   spearman |   n |
|:-------------------------------------|-----------:|----:|
| scout_growth_stage_last              |      0.654 |  13 |
| scout_growth_stage_max               |      0.654 |  13 |
| scout_condition_score_last           |      0.588 |  16 |
| scout_condition_score_min            |      0.58  |  16 |
| scout_condition_not_good_n           |     -0.475 |  21 |
| wx_sow_water_balance                 |      0.461 |  42 |
| wx_sow_et0_sum                       |     -0.429 |  42 |
| wx_sow_dry_days                      |     -0.423 |  42 |
| wx_sow_vpd_days_high                 |     -0.407 |  42 |
| wx_jul_et0                           |     -0.404 |  59 |
| soil_sample_mean_soil_organic_matter |     -0.402 |  36 |
| soil_sample_mean_soil_N              |      0.381 |   8 |

## 4. Leakage rescan (|Spearman| >= 0.9 = suspicious)

No feature has |Spearman| >= 0.9 with target. Clean.

## 5. Label-noise probe (near-identical NDVI/weather, very different yield)

Found **23** near-identical-feature pairs (z-dist<0.6) with yield gap >1 t/ha.

| crop          |   row_i |   row_j |   feat_dist |   yield_i |   yield_j |   gap |
|:--------------|--------:|--------:|------------:|----------:|----------:|------:|
| sunflower     |      91 |     105 |        0.41 |      1    |      3.26 |  2.25 |
| barley_spring |     116 |     122 |        0.6  |      2.27 |      4.5  |  2.23 |
| sunflower     |      13 |      21 |        0.21 |      1.95 |      3.96 |  2.01 |
| barley_spring |     161 |     167 |        0.58 |      1.53 |      3.53 |  2    |
| barley_spring |     162 |     167 |        0.57 |      1.53 |      3.53 |  2    |
| sunflower     |      17 |      21 |        0.41 |      1.98 |      3.96 |  1.98 |
| wheat_winter  |      48 |      49 |        0.2  |      1.3  |      3.28 |  1.98 |
| wheat_spring  |      41 |      45 |        0.57 |      0.38 |      2.34 |  1.96 |
| wheat_winter  |     126 |     127 |        0.24 |      1.5  |      3.44 |  1.94 |
| sunflower     |      90 |     105 |        0.44 |      1.48 |      3.26 |  1.78 |
| wheat_winter  |      81 |      82 |        0.58 |      3.65 |      5.39 |  1.74 |
| wheat_spring  |      92 |     103 |        0.3  |      2.86 |      4.56 |  1.71 |
| barley_spring |      53 |      55 |        0.12 |      2.02 |      3.53 |  1.5  |
| wheat_spring  |     100 |     103 |        0.34 |      3.08 |      4.56 |  1.49 |
| wheat_spring  |      93 |     101 |        0.47 |      3.13 |      1.69 |  1.43 |

(High count = irreducible noise: identical signals map to different yields -> caps R2.)

## 6. Sample sizes per crop x year

| standard_name        |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |
|:---------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| barley_spring        |      0 |      0 |      0 |      3 |      0 |      8 |      1 |      3 |
| maize                |      2 |      0 |      1 |      0 |      0 |      0 |      0 |      0 |
| oil_seed_raps_spring |      0 |      1 |      1 |      0 |      0 |      0 |      0 |      0 |
| oil_seed_raps_winter |      1 |      0 |      3 |      0 |      0 |      0 |      0 |      0 |
| pea                  |      0 |      0 |      0 |      0 |      0 |      1 |      0 |      3 |
| soya                 |      2 |      2 |      0 |      0 |      0 |      0 |      0 |      0 |
| sunflower            |      3 |     14 |      0 |     17 |      3 |      4 |     10 |      8 |
| wheat_spring         |      0 |      1 |     14 |      2 |     17 |      3 |      7 |     10 |
| wheat_winter         |      0 |      3 |      5 |      8 |      6 |      7 |      5 |      0 |

Cells with 1-3 samples make per-crop-year stats and R2 unstable. This is the structural reason metrics swing.
