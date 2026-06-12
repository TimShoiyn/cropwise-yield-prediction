# v21 domain-aligned transfer + honest metrics

NDVI level features shifted into peer space using past-years stats only (leakage-safe). Peer training restricted to our crops. `wcy_r2` = within-crop-year R2 (honest field-ranking skill).

| scenario       | asof   |   n |   aligned_years |   v21_mape |   v21_r2 |   v21_wcy_r2 |   v18raw_mape |   v18raw_r2 |   v18raw_wcy_r2 |   cw_mape |   cw_r2 |   cw_wcy_r2 |
|:---------------|:-------|----:|----------------:|-----------:|---------:|-------------:|--------------:|------------:|----------------:|----------:|--------:|------------:|
| sunflower      | 1 Jul  |  59 |               5 |     23.118 |   -0.079 |       -0.005 |        23.739 |      -0.069 |          -0.025 |    26.147 |  -0.477 |      -0.338 |
| sunflower      | 1 Aug  |  59 |               5 |     23.869 |   -0.459 |        0.066 |        24.459 |      -0.411 |           0.043 |    24.411 |  -0.174 |      -0.242 |
| sunflower      | 1 Sep  |  59 |               5 |     24     |   -0.718 |        0.007 |        27.997 |      -1.081 |          -0.021 |    21.866 |  -0     |      -0.084 |
| wheat_combined | 1 Jul  |  88 |               5 |    131.158 |    0.119 |       -0.013 |       126.918 |       0.212 |           0.049 |    96.228 |   0.5   |      -0.087 |
| wheat_combined | 1 Aug  |  88 |               5 |    128.453 |    0.165 |        0.089 |       122.189 |       0.303 |           0.084 |    85.883 |   0.544 |       0.035 |
| wheat_combined | 1 Sep  |  88 |               5 |     99.387 |    0.451 |        0.023 |        96.54  |       0.507 |           0.055 |    86.232 |   0.539 |      -0.059 |
| all_crops      | 1 Jul  | 180 |               6 |     92.746 |    0.085 |        0.037 |        95.947 |       0.095 |           0.083 |    78.571 |   0.11  |      -0.124 |
| all_crops      | 1 Aug  | 180 |               6 |     74.188 |    0.382 |        0.058 |        75.445 |       0.272 |           0.011 |    68.542 |   0.441 |       0.006 |
| all_crops      | 1 Sep  | 180 |               6 |     71.513 |    0.347 |        0.016 |        68.849 |       0.286 |          -0.007 |    67.838 |   0.432 |      -0.071 |
