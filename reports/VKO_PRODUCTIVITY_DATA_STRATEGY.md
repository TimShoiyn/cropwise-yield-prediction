# VKO productivity_data strategy and v12 result

## What we found

`productivity_data.csv` is not useless. It contains:

- 530 unique VKO field names;
- year;
- crop name in Russian;
- factual yield in `ц/га`;
- forecast yield in `ц/га`;
- active flag.

All 30 fields from `fields.csv` match exactly by normalized Russian field name.

This means we can identify the current 30 fields reliably, but we still cannot identify geometries for the other ~500 fields because no local CSV contains `field_id + name + polygon` for them.

## Critical target bug found

`productivity_data.csv` exposed remaining unit errors in `targets_factual_t_ha.csv`.

Example:

| field_id | field | year | crop | productivity_data fact | previous target |
|---:|---|---:|---|---:|---:|
| 213 | Лесополосы (Козлов увал левый) | 2020 | wheat_spring | 0.578 t/ha | 5.780 t/ha |
| 227 | Большая Семёниха СК | 2020 | wheat_winter | 0.565 t/ha | 5.650 t/ha |
| 208 | Карелкино | 2020 | wheat_spring | 0.505 t/ha | 5.050 t/ha |

Old normalization divided by 10 only when the value exceeded a crop cap. That misses catastrophic low-yield rows stored in `ц/га`.

## v12

Created:

- `scripts/audit/audit_productivity_data_mapping.py`
- `scripts/asof/build_v12_productivity_target.py`
- `scripts/asof/train_compare_asof_v12.py`
- `reports/PRODUCTIVITY_DATA_MAPPING_AUDIT.md`
- `reports/asof_v12_results.md`

v12 uses `productivity_data` factual yield as target for exact-matched 30 fields:

`target_yield_t_ha = урожайность факт ц/га / 10`

Important: `prod_forecast_t_ha`, old target columns, and target deltas are **not** included as ML features.

## Clean v12 results

| scenario | 1 Jul ML | 1 Jul Cropwise | 1 Aug ML | 1 Aug Cropwise | 1 Sep ML | 1 Sep Cropwise |
|---|---:|---:|---:|---:|---:|---:|
| sunflower | **24.0%** | 31.8% | 24.6% | **24.5%** | 25.2% | **17.2%** |
| wheat_combined | **27.4%** | 30.9% | 28.1% | **27.4%** | 29.7% | **29.2%** |
| all_crops | **29.3%** | 29.8% | **27.5%** | 28.9% | **26.9%** | 27.2% |

## Interpretation

This is the real current state:

- We can beat or match Cropwise on all-crops with the corrected target.
- Sunflower: ML wins early, Cropwise wins late.
- Wheat: ML wins on 1 July, Cropwise slightly wins later.
- The model is no longer obviously broken; the main previous issue was target units.
- Remaining bottleneck: only 30 fields have geometry/NDVI/weather coverage.

## Next move

No-token path:

1. Make v12 the new baseline.
2. Rebuild final comparison/discussion around corrected target.
3. Do error analysis on v12 only.
4. Build a separate non-NDVI regional benchmark using all 530 `productivity_data` field names (`field-name/year/crop/forecast`), but do not mix it with remote-sensing ML unless geometries become available.

If token/API becomes available later:

- download polygons for the 500+ additional field names;
- then rebuild NDVI/weather for the full 530-field dataset.
