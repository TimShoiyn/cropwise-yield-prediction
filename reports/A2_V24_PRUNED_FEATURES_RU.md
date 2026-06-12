# A2 v24 pruned feature evaluation

Target policy: `target >= 1 t/ha`. Cropwise benchmark is evaluated on identical OOF rows.

| scenario       | asof   | model               |    n |   mape |     r2 |   wcy_r2 |   rmse |   mae |
|:---------------|:-------|:--------------------|-----:|-------:|-------:|---------:|-------:|------:|
| all_crops      | 1 Jul  | v24_pruned_full     | 1531 | 33.577 |  0.213 |    0.119 |  1.24  | 0.814 |
| all_crops      | 1 Jul  | v24_pruned_no_field | 1531 | 33.281 |  0.203 |    0.113 |  1.247 | 0.813 |
| all_crops      | 1 Jul  | cropwise            | 1497 | 28.74  |  0.215 |    0.009 |  1.245 | 0.756 |
| all_crops      | 1 Aug  | v24_pruned_full     | 1531 | 32.059 |  0.19  |    0.096 |  1.258 | 0.8   |
| all_crops      | 1 Aug  | v24_pruned_no_field | 1531 | 32.163 |  0.163 |    0.074 |  1.279 | 0.809 |
| all_crops      | 1 Aug  | cropwise            | 1497 | 27.355 |  0.289 |    0.071 |  1.184 | 0.686 |
| all_crops      | 1 Sep  | v24_pruned_full     | 1531 | 31.749 |  0.215 |    0.109 |  1.238 | 0.783 |
| all_crops      | 1 Sep  | v24_pruned_no_field | 1531 | 31.697 |  0.22  |    0.127 |  1.235 | 0.789 |
| all_crops      | 1 Sep  | cropwise            | 1497 | 27.08  |  0.291 |    0.06  |  1.183 | 0.681 |
| wheat_combined | 1 Jul  | v24_pruned_full     |  719 | 31.064 |  0.158 |    0.084 |  1.491 | 0.868 |
| wheat_combined | 1 Jul  | v24_pruned_no_field |  719 | 31.268 |  0.151 |    0.083 |  1.497 | 0.875 |
| wheat_combined | 1 Jul  | cropwise            |  698 | 28.273 |  0.153 |    0.009 |  1.509 | 0.801 |
| wheat_combined | 1 Aug  | v24_pruned_full     |  719 | 31.581 |  0.165 |    0.09  |  1.485 | 0.857 |
| wheat_combined | 1 Aug  | v24_pruned_no_field |  719 | 31.188 |  0.148 |    0.078 |  1.5   | 0.862 |
| wheat_combined | 1 Aug  | cropwise            |  698 | 27.463 |  0.201 |    0.03  |  1.466 | 0.741 |
| wheat_combined | 1 Sep  | v24_pruned_full     |  719 | 30.872 |  0.166 |    0.063 |  1.484 | 0.847 |
| wheat_combined | 1 Sep  | v24_pruned_no_field |  719 | 30.556 |  0.124 |    0.034 |  1.521 | 0.855 |
| wheat_combined | 1 Sep  | cropwise            |  698 | 28.444 |  0.188 |    0.011 |  1.478 | 0.763 |
| sunflower      | 1 Jul  | v24_pruned_full     |  372 | 25.377 |  0.056 |    0.081 |  0.746 | 0.59  |
| sunflower      | 1 Jul  | v24_pruned_no_field |  372 | 25.284 |  0.057 |    0.113 |  0.746 | 0.591 |
| sunflower      | 1 Jul  | cropwise            |  367 | 26.508 | -0.229 |   -0.071 |  0.835 | 0.674 |
| sunflower      | 1 Aug  | v24_pruned_full     |  372 | 23.594 |  0.131 |    0.149 |  0.716 | 0.56  |
| sunflower      | 1 Aug  | v24_pruned_no_field |  372 | 23.899 |  0.131 |    0.153 |  0.716 | 0.562 |
| sunflower      | 1 Aug  | cropwise            |  367 | 23.399 |  0.106 |    0.14  |  0.712 | 0.571 |
| sunflower      | 1 Sep  | v24_pruned_full     |  372 | 24.764 |  0.095 |    0.093 |  0.73  | 0.583 |
| sunflower      | 1 Sep  | v24_pruned_no_field |  372 | 24.729 |  0.093 |    0.095 |  0.731 | 0.586 |
| sunflower      | 1 Sep  | cropwise            |  367 | 21.158 |  0.247 |    0.222 |  0.653 | 0.511 |
