# v18 peers dataset summary

Source: `data_raw/productivity_estimate_peers.csv`.

- Target: `productivity / 10` => t/ha.
- `strict`: crop/area/previous crop/sowing + NDVI truncated to as-of.
- `plus`: strict + `accumulated_precipitations`, `average_soil_moisture` (exploratory, horizon not documented).
- Raw `max_ndvi` is intentionally excluded because it can be full-season leakage.

| variant   |   asof_tag |   rows | years     |   crops |   target_mean |   target_min |   target_max | top_crops                                                                                                                                                    |
|:----------|-----------:|-------:|:----------|--------:|--------------:|-------------:|-------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| strict    |      07_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
| plus      |      07_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
| strict    |      08_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
| plus      |      08_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
| strict    |      09_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
| plus      |      09_01 |  26822 | 2010-2025 |      15 |         2.517 |        0.565 |         40.1 | {'sunflower': 13948, 'wheat_spring': 6179, 'barley_spring': 1861, 'wheat_winter': 1465, 'oil_seed_raps_spring': 1013, 'pea': 791, 'soya': 496, 'linum': 415} |
