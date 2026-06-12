# v26 harvest-truth results (Roadmap B3)

Target = real combine harvest (475 fields, 2021-2025), target >= 1 t/ha.
Benchmark = Cropwise productivity estimate on identical rows.
Validation = walk-forward by year. Leakage guard on harvest-time fields.

| scenario       | asof   | model        |   n |    mape |      r2 |   wcy_r2 |    rmse |     mae |
|:---------------|:-------|:-------------|----:|--------:|--------:|---------:|--------:|--------:|
| all_crops      | 1 Jul  | v26_full     | 954 |  44.242 |   0.112 |    0.12  |   1.124 |   0.919 |
| all_crops      | 1 Jul  | v26_no_field | 954 |  43.607 |   0.104 |    0.123 |   1.129 |   0.921 |
| all_crops      | 1 Jul  | cropwise     | 951 |  25.826 |   0.506 |    0.213 |   0.838 |   0.645 |
| all_crops      | 1 Aug  | v26_full     | 906 |  35.673 |   0.264 |    0.191 |   0.984 |   0.791 |
| all_crops      | 1 Aug  | v26_no_field | 906 |  38.138 |   0.204 |    0.185 |   1.023 |   0.83  |
| all_crops      | 1 Aug  | cropwise     | 903 |  26.275 |   0.461 |    0.204 |   0.841 |   0.648 |
| all_crops      | 1 Sep  | v26_full     | 635 |  37.257 |   0.007 |    0.148 |   1.015 |   0.834 |
| all_crops      | 1 Sep  | v26_no_field | 635 |  37.173 |   0.014 |    0.15  |   1.011 |   0.83  |
| all_crops      | 1 Sep  | cropwise     | 632 |  27.193 |   0.262 |    0.108 |   0.873 |   0.664 |
| wheat_combined | 1 Jul  | v26_full     | 495 |  32.64  |   0.34  |    0.144 |   1.008 |   0.819 |
| wheat_combined | 1 Jul  | v26_no_field | 495 |  34.713 |   0.267 |    0.123 |   1.063 |   0.865 |
| wheat_combined | 1 Jul  | cropwise     | 495 |  25.699 |   0.487 |    0.159 |   0.888 |   0.693 |
| wheat_combined | 1 Aug  | v26_full     | 457 |  33.822 |   0.306 |    0.176 |   0.972 |   0.794 |
| wheat_combined | 1 Aug  | v26_no_field | 457 |  35.877 |   0.265 |    0.172 |   1.001 |   0.826 |
| wheat_combined | 1 Aug  | cropwise     | 457 |  26.374 |   0.413 |    0.15  |   0.894 |   0.7   |
| wheat_combined | 1 Sep  | v26_full     | 344 |  37.303 |   0.01  |    0.148 |   1.108 |   0.917 |
| wheat_combined | 1 Sep  | v26_no_field | 344 |  37.255 |   0.044 |    0.164 |   1.089 |   0.902 |
| wheat_combined | 1 Sep  | cropwise     | 344 |  27.709 |   0.275 |    0.074 |   0.948 |   0.738 |
| sunflower      | 1 Jul  | v26_full     | 210 |  23.267 |   0.101 |    0.11  |   0.716 |   0.548 |
| sunflower      | 1 Jul  | v26_no_field | 210 |  23.172 |   0.102 |    0.118 |   0.716 |   0.554 |
| sunflower      | 1 Jul  | cropwise     | 210 |  19.815 |   0.168 |    0.232 |   0.689 |   0.521 |
| sunflower      | 1 Aug  | v26_full     | 210 |  22.248 |   0.161 |    0.16  |   0.692 |   0.535 |
| sunflower      | 1 Aug  | v26_no_field | 210 |  22.147 |   0.149 |    0.16  |   0.697 |   0.542 |
| sunflower      | 1 Aug  | cropwise     | 210 |  19.815 |   0.168 |    0.232 |   0.689 |   0.521 |
| sunflower      | 1 Sep  | v26_full     | 210 |  23.792 |   0.085 |    0.074 |   0.722 |   0.561 |
| sunflower      | 1 Sep  | v26_no_field | 210 |  23.756 |   0.081 |    0.086 |   0.724 |   0.565 |
| sunflower      | 1 Sep  | cropwise     | 210 |  19.815 |   0.168 |    0.232 |   0.689 |   0.521 |
| barley         | 1 Jul  | v26_full     |  84 |  42.107 |   0.052 |    0.116 |   1.382 |   1.171 |
| barley         | 1 Jul  | v26_no_field |  84 |  40.764 |   0.102 |    0.143 |   1.345 |   1.144 |
| barley         | 1 Jul  | cropwise     |  84 |  34.742 |   0.377 |    0.381 |   1.12  |   0.899 |
| barley         | 1 Aug  | v26_full     |  84 |  40.961 |   0.057 |    0.172 |   1.378 |   1.156 |
| barley         | 1 Aug  | v26_no_field |  84 |  39.533 |   0.11  |    0.197 |   1.339 |   1.13  |
| barley         | 1 Aug  | cropwise     |  84 |  34.742 |   0.377 |    0.381 |   1.12  |   0.899 |
| barley         | 1 Sep  | v26_full     |   0 | nan     | nan     |  nan     | nan     | nan     |
| barley         | 1 Sep  | v26_no_field |   0 | nan     | nan     |  nan     | nan     | nan     |
| barley         | 1 Sep  | cropwise     |   0 | nan     | nan     |  nan     | nan     | nan     |
