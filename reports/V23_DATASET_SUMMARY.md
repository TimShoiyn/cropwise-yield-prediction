# v23 Cropwise full dataset summary

Source: refreshed Cropwise API pull (`fields`, `soil_tests`, `historical_values`).

|   asof_tag |   rows |   fields | years     |   crops |   soil_fields |   target_mean | top_crops                                                                                                                                           |
|-----------:|-------:|---------:|:----------|--------:|--------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------|
|      07_01 |   1903 |      483 | 2017-2025 |      14 |           360 |       2.58676 | {'wheat_spring': 698, 'sunflower': 522, 'wheat_winter': 214, 'barley_spring': 172, 'pea': 94, 'soya': 54, 'oil_seed_raps_spring': 38, 'lentil': 35} |
|      08_01 |   1903 |      483 | 2017-2025 |      14 |           360 |       2.58676 | {'wheat_spring': 698, 'sunflower': 522, 'wheat_winter': 214, 'barley_spring': 172, 'pea': 94, 'soya': 54, 'oil_seed_raps_spring': 38, 'lentil': 35} |
|      09_01 |   1903 |      483 | 2017-2025 |      14 |           360 |       2.58676 | {'wheat_spring': 698, 'sunflower': 522, 'wheat_winter': 214, 'barley_spring': 172, 'pea': 94, 'soya': 54, 'oil_seed_raps_spring': 38, 'lentil': 35} |

Target policy: v17-compatible strict target; conflict rows dropped.
Satellite products used: `ndvi`, `temperature`, `soil_moisture`.
