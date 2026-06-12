# Deep Data Audit — Part 2 (suspect investigations)

## A. Field 195 reality check (18 years of sunflower?)
Field 195 — n records 31

```
 year        standard_name  productivity  harvested_weight sowing_date harvesting_date
 2000            sunflower           NaN               NaN         NaN             NaN
 2001            sunflower           NaN               NaN         NaN             NaN
 2002            sunflower           NaN               NaN         NaN             NaN
 2003            sunflower           NaN               NaN         NaN             NaN
 2004            sunflower           NaN               NaN         NaN             NaN
 2005            sunflower           NaN               NaN         NaN             NaN
 2006            sunflower           NaN               NaN         NaN             NaN
 2007            sunflower           NaN               NaN         NaN             NaN
 2008            sunflower           NaN               NaN         NaN             NaN
 2009            sunflower           NaN               NaN         NaN             NaN
 2010            sunflower           NaN               NaN         NaN             NaN
 2011            sunflower           NaN               NaN         NaN             NaN
 2012            sunflower           NaN               NaN         NaN             NaN
 2013            sunflower           NaN               NaN         NaN             NaN
 2014            sunflower           NaN               NaN         NaN             NaN
 2015            sunflower           NaN               NaN         NaN             NaN
 2016            sunflower           NaN               NaN         NaN             NaN
 2017            sunflower           NaN               NaN         NaN             NaN
 2018            sunflower           NaN               NaN         NaN             NaN
 2019         wheat_spring           NaN               NaN         NaN             NaN
 2020 oil_seed_raps_spring           NaN               NaN         NaN             NaN
 2021        barley_spring     20.222764            248.74  2021-05-12      2021-08-22
 2022         wheat_winter     21.216260            260.96  2021-09-08      2022-08-10
 2023         wheat_winter      8.904065            109.52  2022-08-29      2023-08-09
 2024            sunflower     27.365854            336.60  2024-05-12      2024-09-29
 2025         wheat_spring     27.195122            334.50  2025-05-14      2025-09-18
 2026        barley_spring           NaN               NaN         NaN             NaN
 2027         wheat_spring           NaN               NaN         NaN             NaN
 2028                  NaN           NaN               NaN         NaN             NaN
 2029                  NaN           NaN               NaN         NaN             NaN
 2030                  NaN           NaN               NaN         NaN             NaN
```

Distinct crops on field 195:

```
standard_name
sunflower               20
wheat_spring             3
barley_spring            2
wheat_winter             2
oil_seed_raps_spring     1
```

Top-10 fields by record count + their crop diversity:

```
           n       years  crops_distinct modal_crop  modal_share_pct
field_id                                                            
195       31  2000..2030               5  sunflower             71.4
431       31  2000..2030               4   sainfoin             62.5
335       31  2000..2030               5  sunflower             76.9
336       31  2000..2030               3  sunflower             79.2
338       31  2000..2030               5  sunflower             77.8
339       31  2000..2030               4  sunflower             76.9
340       31  2000..2030               5  sunflower             76.9
341       31  2000..2030               5  sunflower             76.9
342       31  2000..2030               6  sunflower             76.9
343       31  2000..2030               6  sunflower             76.0
```

## B. yield_maps.csv inspection
rows: 104, columns: 34

```
id, field_id, additional_info, external_id, description, field_work_result_id, property_name, calculated_average, units, created_at, updated_at, external_average, grid_map.geotransform.top_left_x, grid_map.geotransform.top_left_y, grid_map.geotransform.step_x, grid_map.geotransform.step_y, grid_map.geotransform.size_x, grid_map.geotransform.size_y, grid_map.data_values, totals.result.area.units, totals.result.area.value, totals.result.total.units, totals.result.total.value, totals.result.average.units, totals.result.average.value, totals.result.fuel_volume.units, totals.result.fuel_volume.value, totals.result.time_effective.units, totals.result.time_effective.value, totals.result.time_ineffective.units, totals.result.time_ineffective.value, totals.result.average_yield_moisture.units, totals.result.average_yield_moisture.value, totals.source
```

Field/year distribution:

```
field_id
195    8
197    4
207    4
208    4
209    4
210    2
211    8
212    4
213    4
214    4
215    4
217    2
219    2
222    4
223    4
224    4
225    2
226    2
227    2
228    4
229    4
230    8
231    2
232    6
233    4
234    4
```

Date span of yield_maps: 2021-09-03 13:33:21.219000+05:00 -> 2024-10-03 07:20:36.066000+05:00

Average yield from yield_maps (top 30 rows):

```
 field_id  totals.result.average.value totals.result.average.units  calculated_average         units  external_average
      195                          NaN                         NaN            1.447266   tonn_per_ha               NaN
      195                          NaN                         NaN           16.404688       percent               NaN
      195                         3.16                 tonn_per_ha            0.000000   tonn_per_ha              3.16
      195                         3.16                 tonn_per_ha           24.400000       percent               NaN
      195                         3.16                 tonn_per_ha            0.000000   tonn_per_ha              3.16
      195                         3.16                 tonn_per_ha           24.400000       percent               NaN
      195                         3.04                 tonn_per_ha            1.080261 tonn_per_acre              3.04
      195                         3.04                 tonn_per_ha            7.410330       percent               NaN
      197                          NaN                         NaN            1.192079   tonn_per_ha               NaN
      197                          NaN                         NaN           15.451501       percent               NaN
      197                         3.48                 tonn_per_ha            0.000000 tonn_per_acre              3.48
      197                         3.48                 tonn_per_ha           19.501872       percent               NaN
      207                          NaN                         NaN            1.094123   tonn_per_ha               NaN
      207                          NaN                         NaN           18.531579       percent               NaN
      207                         3.41                 tonn_per_ha            0.602185 tonn_per_acre              3.41
      207                         3.41                 tonn_per_ha            9.116098       percent               NaN
      208                          NaN                         NaN            2.593432   tonn_per_ha               NaN
      208                          NaN                         NaN           11.487574       percent               NaN
      208                         2.42                 tonn_per_ha            0.664614 tonn_per_acre              2.42
      208                         2.42                 tonn_per_ha            6.821399       percent               NaN
      209                          NaN                         NaN            2.671014   tonn_per_ha               NaN
      209                          NaN                         NaN           14.744658       percent               NaN
      209                         3.31                 tonn_per_ha            0.561250   tonn_per_ha              3.31
      209                         3.31                 tonn_per_ha           13.712500       percent               NaN
      210                          NaN                         NaN            2.922967   tonn_per_ha               NaN
      210                          NaN                         NaN           10.933234       percent               NaN
      211                          NaN                         NaN            0.000000   tonn_per_ha               NaN
      211                          NaN                         NaN            6.318317       percent               NaN
      211                          NaN                         NaN            0.000000   tonn_per_ha               NaN
      211                          NaN                         NaN           13.800000       percent               NaN
```

## C. productivity_data.csv inspection
rows: 3070, columns: 7

```
Поле, Регион, Год, Культура, урожайность факт ц/га, урожайность прогноз ц/га, active
```

First 10 rows:

```
              Поле Регион  Год       Культура  урожайность факт ц/га  урожайность прогноз ц/га  active
         1-я речка    ВКО 2018   Подсолнечник              28.040000                    28.237    True
         1-я речка    ВКО 2019 Пшеница яровая              40.830000                    38.867    True
         1-я речка    ВКО 2020    Рапс яровой               7.700000                    15.200    True
         1-я речка    ВКО 2021  Ячмень яровой              30.677778                    34.913    True
         1-я речка    ВКО 2022 Пшеница озимая              20.529630                    26.808    True
         1-я речка    ВКО 2023 Пшеница озимая              10.088889                    21.881    True
         1-я речка    ВКО 2024   Подсолнечник              35.918519                    30.239    True
         1-я речка    ВКО 2025 Пшеница яровая              44.196296                    43.208    True
2 Коновалиха Серов    ВКО 2025 Пшеница яровая              48.266667                    44.002    True
     2-я речка лев    ВКО 2018    Рапс озимый              10.000000                    16.267    True
```

Dtypes:

```
Поле                         object
Регион                       object
Год                           int64
Культура                     object
урожайность факт ц/га       float64
урожайность прогноз ц/га    float64
active                         bool
```

## D. plant_threats + field_scout_reports
plant_threats.csv rows: 918
Note: this looks like a static catalog (threats master list), not per-field events.
Distinct threat_type: {'insect': 338, 'weed': 276, 'disease': 256, 'nutrition_problem': 20, 'other': 14, 'damaged_area': 10, 'technological_mistake': 4}

field_scout_reports rows: 2636
Field/season coverage:

```
 field_id  season  n
      195    2019  1
      195    2021  1
      195    2022  1
      195    2024  1
      197    2019  1
      197    2021  1
      197    2022  1
      197    2023  4
      197    2024  2
      197    2025 24
      207    2021  2
      207    2022  2
      207    2023  3
      207    2024  3
      208    2022  2
      208    2023  1
      208    2024  1
      209    2020  3
      209    2022  2
      209    2023  2
      209    2024  3
      209    2025  1
      210    2022  2
      210    2023  3
      210    2024  4
      210    2025  3
      212    2021  1
      212    2022  2
      212    2023  1
      212    2024  1
      213    2022  2
      213    2023  1
      214    2022  2
      214    2023  1
      215    2021  1
      215    2022  2
      215    2023  1
      215    2024  3
      216    2021  2
      216    2022  2
```

Report time span: 2019-11-07 16:02:21.273000+05:00 -> 2026-01-23 13:53:19.377000+05:00
Reports per year:

```
year
2019      2
2020      5
2021    232
2022    652
2023    680
2024    591
2025    469
2026      5
```

Top growth_stage values:

```
growth_stage
75    53
59    38
77    33
22    28
29    28
16    27
51    27
14    27
21    25
12    24
92    24
83    23
69    20
15    19
13    19
23    19
89    17
85    16
87    15
39    14
```

Reports with non-empty threats field: 8.3%
Sample threats values:

```
16    [{'id': 11, 'name': 'Снижение густоты растений...
18    [{'id': 2, 'name': 'Тип ущерба не определен', ...
19    [{'id': 5, 'name': 'Вымерзание растений', 'typ...
21    [{'id': 3, 'name': 'Просев', 'type': 'Технолог...
26    [{'id': 7, 'name': 'Вымочка', 'type': 'Зоны ущ...
27    [{'id': 6, 'name': 'Пропуск', 'type': 'Техноло...
30    [{'id': 9, 'name': 'Вымочка', 'type': 'Зоны ущ...
31    [{'id': 8, 'name': 'Пропуск', 'type': 'Техноло...
32    [{'id': 10, 'name': 'Росичка кровоостанавливаю...
36    [{'id': 12, 'name': 'Вымочка', 'type': 'Зоны у...
```

## E. NDVI cloud_coverage actually filtered?
rows: 757853

cloud_coverage distribution:

```
count    0.0
mean     NaN
std      NaN
min      NaN
5%       NaN
25%      NaN
50%      NaN
75%      NaN
90%      NaN
95%      NaN
99%      NaN
max      NaN
```

data_coverage distribution:

```
count    0.0
mean     NaN
std      NaN
min      NaN
5%       NaN
25%      NaN
50%      NaN
75%      NaN
90%      NaN
95%      NaN
99%      NaN
max      NaN
```

Field-level NDVI Jun-Aug median (cloud-clean):

```
          median   mean    std
field_id                      
229        0.522  0.515  0.090
195        0.530  0.515  0.159
222        0.545  0.539  0.145
207        0.549  0.523  0.163
233        0.551  0.544  0.127
228        0.555  0.544  0.139
231        0.556  0.536  0.152
223        0.559  0.551  0.134
217        0.560  0.546  0.141
218        0.562  0.540  0.130
227        0.563  0.532  0.159
232        0.566  0.535  0.162
211        0.576  0.555  0.134
219        0.577  0.548  0.157
215        0.579  0.569  0.114
212        0.580  0.545  0.145
209        0.581  0.554  0.145
230        0.583  0.558  0.145
208        0.587  0.578  0.121
197        0.587  0.554  0.153
214        0.588  0.572  0.115
213        0.588  0.572  0.114
226        0.588  0.579  0.109
224        0.597  0.587  0.115
220        0.601  0.597  0.102
210        0.604  0.571  0.142
234        0.624  0.599  0.106
216        0.643  0.628  0.091
225        0.649  0.644  0.079
221        0.656  0.647  0.082
```

## F. Cropwise estimate accuracy by year
Cropwise estimate accuracy by year:

```
         n   bias    mae   rmse  mape_%
year                                   
2017  17.0  0.119  0.644  0.790  44.843
2018   8.0  0.383  0.831  1.262  37.219
2019  21.0 -0.070  0.553  0.740  22.559
2020  24.0 -1.211  1.723  2.128  64.274
2021  30.0  0.393  0.678  0.852  32.088
2022  26.0  0.398  0.614  0.765  30.475
2023  24.0 -0.061  0.720  0.916  39.767
2024  24.0  0.145  0.510  0.623  16.188
2025  26.0  0.226  0.683  0.866  33.734
```

Cropwise per (year, crop):

```
                              n   mae  mape_%
year standard_name                           
2017 oil_seed_raps_spring   9.0  0.62   58.06
     sunflower              2.0  0.62   18.64
     wheat_spring           2.0  0.54   27.72
     wheat_winter           4.0  0.76   36.77
2018 maize                  2.0  1.80   54.37
     oil_seed_raps_winter   1.0  0.76   77.49
     soya                   2.0  0.17   18.30
     sunflower              3.0  0.65   24.97
2019 oil_seed_raps_spring   1.0  0.76   65.19
     soya                   2.0  0.49   33.34
     sunflower             14.0  0.56   20.43
     wheat_spring           1.0  0.81   22.55
     wheat_winter           3.0  0.42   11.11
2020 maize                  1.0  2.24  386.89
     oil_seed_raps_spring   1.0  3.21   73.28
     oil_seed_raps_winter   3.0  0.92   32.47
     wheat_spring          14.0  1.94   57.88
     wheat_winter           5.0  1.21   34.93
2021 barley_spring          3.0  1.01   43.60
     sunflower             17.0  0.49   23.77
     wheat_spring           2.0  1.67  118.63
     wheat_winter           8.0  0.71   23.81
2022 sunflower              3.0  0.74   63.28
     wheat_spring          17.0  0.52   19.07
     wheat_winter           6.0  0.81   46.39
2023 barley_spring          9.0  0.73   27.83
     pea                    1.0  0.07    4.34
     sunflower              4.0  0.19    7.22
     wheat_spring           3.0  0.76   41.12
     wheat_winter           7.0  1.08   78.20
2024 barley_spring          1.0  0.57   18.95
     sunflower             10.0  0.53   20.06
     wheat_spring           7.0  0.61   17.65
     wheat_winter           6.0  0.36    7.57
2025 barley_spring          3.0  1.12   69.19
     pea                    3.0  1.02   67.88
     sunflower              8.0  0.37   12.85
     wheat_spring          11.0  0.76   32.94
```

## G. v7 / v9 dataset coverage stats

v7 07_01: rows 201, fields 30, years 2017..2025

v7 08_01: rows 201, fields 30, years 2017..2025

v7 09_01: rows 201, fields 30, years 2017..2025

v9 07_01: rows 201, fields 30, years 2017..2025

v9 08_01: rows 201, fields 30, years 2017..2025

v9 09_01: rows 201, fields 30, years 2017..2025

## H. soil_tests (179 numeric features) coverage
rows: 680, cols: 182
distinct fields with soil tests: 421
made_at span: 2020-01-01 00:00:00 -> 2024-10-20 00:00:00

```
 field_id  n_tests
      195        2
      197        2
      207        2
      208        3
      209        2
      210        2
      211        1
      212        2
      213        1
      214        1
      215        2
      216        2
      217        2
      218        2
      219        2
      220        2
      222        2
      223        1
      224        2
      226        2
      227        2
      228        2
      229        2
      230        2
      231        2
      232        2
      233        2
      234        1
      235        1
      236        2
      237        2
      238        2
      239        2
      241        2
      242        2
      244        1
      246        2
      247        2
      248        2
      249        1
      250        2
      251        2
      252        2
      253        2
      254        2
      255        2
      256        2
      257        2
      258        2
      259        3
      260        2
      261        2
      262        2
      263        1
      264        1
      265        1
      266        1
      267        1
      268        1
      269        2
      270        2
      271        2
      272        2
      273        2
      274        2
      276        2
      277        2
      278        2
      279        2
      280        2
      282        2
      283        2
      284        1
      286        1
      287        1
      288        1
      289        1
      291        1
      292        2
      295        2
      296        2
      297        1
      298        2
      299        2
      300        1
      301        2
      302        2
      303        2
      304        2
      305        2
      306        2
      307        2
      308        2
      309        1
      310        2
      311        2
      312        3
      313        2
      314        2
      315        3
      316        1
      317        2
      318        2
      319        2
      320        1
      321        3
      322        2
      323        2
      324        2
      325        2
      327        2
      328        2
      329        1
      330        2
      331        2
      332        2
      333        1
      334        1
      335        1
      336        1
      338        1
      339        1
      340        2
      341        2
      342        2
      343        2
      344        2
      345        1
      346        2
      347        2
      349        2
      350        2
      352        2
      356        2
      357        1
      358        2
      359        2
      360        1
      361        2
      362        2
      363        2
      364        2
      365        2
      366        1
      367        2
      368        2
      369        2
      370        1
      371        2
      372        2
      373        2
      374        1
      375        1
      376        1
      377        2
      378        2
      379        3
      380        1
      381        2
      382        2
      383        1
      384        2
      386        2
      387        2
      388        2
      389        1
      390        2
      391        2
      392        2
      393        3
      394        1
      395        2
      396        2
      397        2
      402        2
      403        2
      405        2
      406        2
      407        1
      408        2
      409        3
      410        1
      411        2
      412        2
      414        2
      415        2
      416        2
      418        2
      419        2
      420        1
      421        2
      422        2
      423        2
      424        2
      425        2
      426        2
      427        2
      428        3
      429        2
      430        1
      432        1
      435        1
      440        1
      457        1
      469        1
      512        1
      513        2
      514        1
      611        3
      612        3
      616        3
      628        2
      630        1
      632        1
      633        1
      634        1
      635        2
      636        1
      640        2
      643        3
      644        1
      649        1
      650        2
      652        1
      653        1
      654        1
      655        2
      657        1
      659        1
      661        2
      662        1
      663        2
      665        1
      666        1
      667        2
      668        2
      669        2
      670        2
      671        1
      672        2
      674        2
      675        1
      677        1
      680        2
      685        2
      689        1
      690        1
      693        1
      694        1
      696        1
      698        1
      699        2
      700        1
      706        1
      711        1
      712        1
      715        1
      716        1
      717        1
      718        2
      722        1
      725        1
      726        1
      727        2
      728        2
      729        1
      730        1
      731        1
      733        2
      736        1
      737        2
      738        2
      741        1
      742        3
      743        1
      744        1
      745        1
      746        1
      747        1
      749        2
      750        1
      752        1
      753        1
      755        1
      756        1
      764        2
      766        1
      767        1
      768        2
      769        2
      770        2
      771        1
      772        2
      773        1
      774        2
      775        2
      776        1
      777        2
      778        1
      779        1
      780        1
      781        1
      782        2
      783        1
      784        2
      785        1
      786        1
      787        2
      788        1
      789        1
      790        3
      791        1
      792        1
      793        1
      794        1
      795        1
      796        1
      797        1
      798        1
      799        2
      800        1
      801        1
      802        1
      803        1
      805        1
      807        2
      808        2
      809        1
      810        1
      811        1
      812        1
      814        1
      815        1
      818        1
      821        2
      823        1
      824        1
      831        1
      832        1
      833        1
      834        1
      836        1
      837        1
      838        1
      839        1
      842        1
      846        1
      847        1
      848        1
      849        1
      850        2
      851        2
      852        2
      874        1
      879        1
      881        1
      892        2
      893        2
      894        2
      895        1
      896        2
      897        2
      898        1
      899        2
      900        2
      901        2
      902        1
      903        2
      904        2
      905        2
      906        2
      907        2
      908        2
      909        2
      910        2
      911        2
      912        2
      913        2
      914        2
      915        1
      916        1
      917        1
      918        2
      919        1
      920        1
      921        2
      922        2
      923        2
      924        2
      925        2
      926        2
      928        1
      929        2
      930        2
      931        2
      932        2
      933        2
      934        2
      935        1
      936        2
      937        2
      938        2
      939        2
      940        1
      941        2
      942        2
      943        2
      944        2
      945        1
      946        2
      947        2
      949        2
      951        1
      957        1
      964        1
      966        1
      968        1
      969        1
      970        1
      972        1
      973        1
```

soil_test_samples rows: 1977, cols: 33
Per-column NaN ratio (top 20 most populated):

```
id                                     0.000
soil_pH                                0.000
soil_S                                 0.000
soil_P                                 0.000
soil_K                                 0.000
soil_organic_matter                    0.000
created_at                             0.000
coordinates                            0.000
soil_test_id                           0.000
updated_at                             0.000
soil_Cu                                0.005
soil_Fe                                0.005
soil_Mg                                0.005
soil_Mn                                0.005
soil_Mo                                0.005
soil_Zn                                0.005
soil_research_method_organic_matter    0.364
soil_research_method_pH                0.364
soil_research_method_S                 0.364
soil_research_method_K                 0.364
```
