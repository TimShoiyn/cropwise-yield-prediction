# v27 — Sentinel-2 ablation on harvest truth

Fair ablation on identical S2-covered, pre-harvest rows.
`v26_base` = no S2; `v27_s2` = + NDRE/EVI/GCVI; `cropwise_asof` = same-date benchmark.

| scenario       | asof   | model         |   n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:-------|:--------------|----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | 1 Jul  | v26_base      | 946 | 46.256 |  0.01  |    0.072 |  1.185 | 0.967 |
| all_crops      | 1 Jul  | v27_s2        | 946 | 41.887 |  0.165 |    0.151 |  1.088 | 0.888 |
| all_crops      | 1 Jul  | cropwise_asof | 943 | 29.348 |  0.333 |    0.025 |  0.972 | 0.751 |
| all_crops      | 1 Aug  | v26_base      | 904 | 42.6   |  0.124 |    0.144 |  1.073 | 0.881 |
| all_crops      | 1 Aug  | v27_s2        | 904 | 41.414 |  0.139 |    0.147 |  1.064 | 0.868 |
| all_crops      | 1 Aug  | cropwise_asof | 901 | 27.829 |  0.44  |    0.18  |  0.857 | 0.666 |
| all_crops      | 1 Sep  | v26_base      |  63 | 22.146 |  0.147 |    0.13  |  0.762 | 0.598 |
| all_crops      | 1 Sep  | v27_s2        |  63 | 22.005 |  0.107 |    0.187 |  0.78  | 0.621 |
| all_crops      | 1 Sep  | cropwise_asof |  63 | 24.713 |  0.083 |    0.068 |  0.79  | 0.629 |
| wheat_combined | 1 Jul  | v26_base      | 490 | 35.703 |  0.273 |    0.112 |  1.06  | 0.864 |
| wheat_combined | 1 Jul  | v27_s2        | 490 | 33.787 |  0.302 |    0.093 |  1.039 | 0.838 |
| wheat_combined | 1 Jul  | cropwise_asof | 490 | 28.367 |  0.338 |    0.014 |  1.012 | 0.772 |
| wheat_combined | 1 Aug  | v26_base      | 456 | 38.977 |  0.165 |    0.166 |  1.066 | 0.893 |
| wheat_combined | 1 Aug  | v27_s2        | 456 | 37.031 |  0.22  |    0.154 |  1.031 | 0.864 |
| wheat_combined | 1 Aug  | cropwise_asof | 456 | 26.44  |  0.454 |    0.198 |  0.862 | 0.676 |
| sunflower      | 1 Jul  | v26_base      | 210 | 23.401 |  0.079 |    0.107 |  0.725 | 0.556 |
| sunflower      | 1 Jul  | v27_s2        | 210 | 23.327 |  0.086 |    0.119 |  0.722 | 0.555 |
| sunflower      | 1 Jul  | cropwise_asof | 210 | 25.657 | -0.335 |    0.003 |  0.873 | 0.71  |
| sunflower      | 1 Aug  | v26_base      | 209 | 22.804 |  0.159 |    0.116 |  0.694 | 0.524 |
| sunflower      | 1 Aug  | v27_s2        | 209 | 23.148 |  0.142 |    0.114 |  0.701 | 0.545 |
| sunflower      | 1 Aug  | cropwise_asof | 209 | 23.653 | -0.029 |    0.111 |  0.768 | 0.611 |
| barley         | 1 Jul  | v26_base      |  83 | 41.988 |  0.071 |    0.113 |  1.364 | 1.146 |
| barley         | 1 Jul  | v27_s2        |  83 | 39.051 |  0.107 |    0.326 |  1.337 | 1.127 |
| barley         | 1 Jul  | cropwise_asof |  83 | 42.438 |  0.106 |    0.116 |  1.337 | 1.109 |
| barley         | 1 Aug  | v26_base      |  84 | 39.459 |  0.116 |    0.193 |  1.335 | 1.129 |
| barley         | 1 Aug  | v27_s2        |  84 | 40.458 |  0.011 |    0.186 |  1.412 | 1.184 |
| barley         | 1 Aug  | cropwise_asof |  84 | 40.149 |  0.242 |    0.245 |  1.236 | 0.986 |
