# v24 crop-specific (agronomic) results

Dataset: v24 agro (soil_moisture fixed + GDD + heat/cold stress), target >= 1 t/ha.

Models: `v24_full` (with field_id), `v24_no_field`, `cropwise` (benchmark, same rows).

| scenario       | asof   | model        |    n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:-------|:-------------|-----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | 1 Jul  | v24_full     | 1531 | 32.736 |  0.233 |    0.143 |  1.224 | 0.8   |
| all_crops      | 1 Jul  | v24_no_field | 1531 | 33.617 |  0.205 |    0.076 |  1.247 | 0.809 |
| all_crops      | 1 Jul  | cropwise     | 1497 | 28.74  |  0.215 |    0.009 |  1.245 | 0.756 |
| all_crops      | 1 Aug  | v24_full     | 1531 | 32.313 |  0.212 |    0.122 |  1.241 | 0.792 |
| all_crops      | 1 Aug  | v24_no_field | 1531 | 33.227 |  0.16  |    0.071 |  1.281 | 0.804 |
| all_crops      | 1 Aug  | cropwise     | 1497 | 27.355 |  0.289 |    0.071 |  1.184 | 0.686 |
| all_crops      | 1 Sep  | v24_full     | 1531 | 32.836 |  0.189 |    0.113 |  1.259 | 0.812 |
| all_crops      | 1 Sep  | v24_no_field | 1531 | 34.087 |  0.182 |    0.101 |  1.264 | 0.815 |
| all_crops      | 1 Sep  | cropwise     | 1497 | 27.08  |  0.291 |    0.06  |  1.183 | 0.681 |
| wheat_combined | 1 Jul  | v24_full     |  719 | 31.099 |  0.168 |    0.072 |  1.482 | 0.869 |
| wheat_combined | 1 Jul  | v24_no_field |  719 | 31.565 |  0.152 |    0.067 |  1.496 | 0.885 |
| wheat_combined | 1 Jul  | cropwise     |  698 | 28.273 |  0.153 |    0.009 |  1.509 | 0.801 |
| wheat_combined | 1 Aug  | v24_full     |  719 | 30.414 |  0.172 |    0.071 |  1.479 | 0.831 |
| wheat_combined | 1 Aug  | v24_no_field |  719 | 30.232 |  0.137 |    0.034 |  1.509 | 0.847 |
| wheat_combined | 1 Aug  | cropwise     |  698 | 27.463 |  0.201 |    0.03  |  1.466 | 0.741 |
| wheat_combined | 1 Sep  | v24_full     |  719 | 33.325 |  0.135 |    0.041 |  1.511 | 0.899 |
| wheat_combined | 1 Sep  | v24_no_field |  719 | 34.196 |  0.128 |    0.018 |  1.518 | 0.913 |
| wheat_combined | 1 Sep  | cropwise     |  698 | 28.444 |  0.188 |    0.011 |  1.478 | 0.763 |
| sunflower      | 1 Jul  | v24_full     |  372 | 24.767 |  0.116 |    0.128 |  0.722 | 0.569 |
| sunflower      | 1 Jul  | v24_no_field |  372 | 24.8   |  0.114 |    0.143 |  0.723 | 0.568 |
| sunflower      | 1 Jul  | cropwise     |  367 | 26.508 | -0.229 |   -0.071 |  0.835 | 0.674 |
| sunflower      | 1 Aug  | v24_full     |  372 | 23.206 |  0.18  |    0.17  |  0.695 | 0.54  |
| sunflower      | 1 Aug  | v24_no_field |  372 | 23.009 |  0.172 |    0.159 |  0.699 | 0.544 |
| sunflower      | 1 Aug  | cropwise     |  367 | 23.399 |  0.106 |    0.14  |  0.712 | 0.571 |
| sunflower      | 1 Sep  | v24_full     |  372 | 23.604 |  0.146 |    0.132 |  0.709 | 0.557 |
| sunflower      | 1 Sep  | v24_no_field |  372 | 23.604 |  0.151 |    0.139 |  0.708 | 0.554 |
| sunflower      | 1 Sep  | cropwise     |  367 | 21.158 |  0.247 |    0.222 |  0.653 | 0.511 |
