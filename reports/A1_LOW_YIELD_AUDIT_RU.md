# A1 low-yield target audit (v24)

Dataset: `ml_dataset_v24_agro_asof_08_01.csv` (1 Aug snapshot).

## Main counts

- Total target rows: **1,903**
- Normal rows (`target >= 1.0`): **1,684** (88.5%)
- Low-yield rows (`target < 1.0`): **219** (11.5%)

## Classification

| low_yield_class                       |   count |
|:--------------------------------------|--------:|
| suspect_target_high_ndvi_and_cropwise |     119 |
| ambiguous_low_yield                   |      52 |
| suspect_target_high_ndvi              |      48 |

Interpretation:

- `likely_crop_failure_low_ndvi`: low target is agronomically plausible; NDVI never formed a strong canopy.
- `suspect_target_high_ndvi*`: target is suspicious; field looked productive by NDVI and/or Cropwise expected a normal crop.
- `ambiguous_low_yield`: not enough evidence to trust or reject; keep separate from main yield metric.

## By crop

| standard_name        |   count |
|:---------------------|--------:|
| wheat_spring         |      56 |
| sunflower            |      41 |
| soya                 |      30 |
| pea                  |      20 |
| wheat_winter         |      19 |
| oil_seed_raps_spring |      15 |
| oil_seed_raps_winter |      14 |
| lentil               |      12 |
| barley_spring        |       5 |
| maize                |       3 |
| buckwheat            |       2 |
| rye_winter           |       1 |

## By year

|   year |   count |
|-------:|--------:|
|   2017 |      14 |
|   2018 |      30 |
|   2019 |      10 |
|   2020 |      89 |
|   2021 |       1 |
|   2022 |       4 |
|   2023 |      21 |
|   2024 |      38 |
|   2025 |      12 |

## Target source among low-yield rows

| target_source         |   count |
|:----------------------|--------:|
| v23_prod_fact_t_ha    |     125 |
| v23_physical_t_ha     |      77 |
| v23_productivity_t_ha |      17 |

## NDVI sanity

`ndvi_max_asof` distribution:

| metric   |   low_yield |      normal |
|:---------|------------:|------------:|
| count    |  219        | 1684        |
| mean     |    0.632963 |    0.718496 |
| p25      |    0.556    |    0.681    |
| median   |    0.647    |    0.7345   |
| p75      |    0.714    |    0.766    |
| max      |    0.813    |    0.837    |

## Decision policy for training/evaluation

1. Main yield-regression benchmark should use `target >= 1 t/ha`.
2. Low-yield rows should not be mixed into MAPE because they represent a different task and can explode percentage error.
3. Keep low-yield rows as a separate **crop failure / anomaly detection** problem.
4. For dissertation claims, report both:
   - main regression on normal-yield rows;
   - separate low-yield audit coverage and examples.

## Key numbers

- Likely real crop failures: **0**
- Suspect target rows with high NDVI evidence: **167**
- Detailed rows: `reports/microscope/a1_low_yield_rows.csv`
- Crop-year summary: `reports/microscope/a1_low_yield_summary_by_crop_year.csv`
