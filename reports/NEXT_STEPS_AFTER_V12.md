# Next steps after v12 — corrected target baseline

## Current truth

v12 is now the main baseline.

Why: `productivity_data.csv` proved that old factual targets still had unit bugs, especially low-yield 2020 rows stored in `ц/га` but interpreted as `t/ha`.

Sanity audit:

- leakage columns: 0;
- duplicate `(field_id, year)`: 0;
- crop-label mismatches between `prod_crop_ru` and `standard_name`: 0;
- target range: `0.168..6.384 t/ha`;
- rows: 183, fields: 30, years: 2018..2025.

Reports:

- `reports/V12_PIPELINE_SANITY_AUDIT.md`
- `reports/V12_ERROR_ANALYSIS.md`
- `reports/asof_v12_results.md`
- `reports/v12_oof_predictions_best.csv`

## v12 benchmark

| scenario | 1 Jul ML | 1 Jul Cropwise | 1 Aug ML | 1 Aug Cropwise | 1 Sep ML | 1 Sep Cropwise |
|---|---:|---:|---:|---:|---:|---:|
| all_crops | **29.3%** | 29.8% | **27.5%** | 28.9% | **26.9%** | 27.2% |
| sunflower | **24.0%** | 31.8% | 24.6% | **24.5%** | 25.2% | **17.2%** |
| wheat_combined | **27.4%** | 30.9% | 28.1% | **27.4%** | 29.7% | **29.2%** |

Important nuance: ML wins MAPE on all-crops, but row-level win rate is only ~38-43%. Cropwise has lower MAE/RMSE in several cases because ML badly misses some high-yield fields.

## New root cause after error analysis

The largest current ML error is not low-yield catastrophe anymore. It is **high-yield underprediction**.

Worst example:

- `field_id=209`, `Козлов Увал левый`, wheat_winter 2024;
- target: `6.384 t/ha`;
- ML prediction: ~`2.7-3.1 t/ha`;
- Cropwise much closer.

This means v12 fixed target units, but the model still lacks a signal that explains unusually high yields.

Likely missing signals:

1. management intensity / inputs;
2. cultivar/variety/seed quality;
3. correct operation timing;
4. yield-map correction vs productivity target;
5. field-specific production potential with too few examples;
6. 2024 weather pattern not represented well by current features.

## Immediate next tasks

### 1. High-yield miss audit

File to create next:

- `scripts/audit/high_yield_miss_audit_v12.py`

For worst ML underpredictions, inspect:

- crop,
- field,
- target source,
- NDVI peak/integral/last,
- weather stress,
- NPK operations,
- scout reports,
- soil samples,
- yield_maps if available.

Goal: find what feature is missing or whether target is still suspicious.

### 2. Operation/features bug audit

Check whether operation features correctly include:

- seed operation rates;
- fertilizer N/P/K by actual completed area;
- herbicide/fungicide/insecticide timing;
- operation completion before as-of date only;
- no post-asof leakage.

This is important because high-yield fields may be explained by management intensity, not NDVI.

### 3. Build v13 management-focused features

Potential v13 features:

- seed_rate_kg_ha or seed_units_ha;
- fertilizer timing before sowing / after sowing;
- N/P/K split applications;
- fungicide_count_asof;
- herbicide_count_asof;
- insecticide_count_asof;
- operation_delay: planned vs actual;
- sowing operation completed date from operations table, not only history sowing_date.

### 4. Regional 530-field non-NDVI benchmark

Use all `productivity_data.csv` rows, but keep it separate from remote-sensing ML:

- field name,
- crop,
- year,
- previous crop from same field name,
- historical field mean,
- year/crop mean,
- Cropwise forecast from same file.

This answers: how much can be predicted from field history alone?

## Rule going forward

Before adding any new feature/model, run:

1. target sanity;
2. leakage scan;
3. crop-label consistency;
4. worst-error inspection;
5. feature as-of validity check.

No more trusting intermediate CSVs blindly.


## High-yield miss audit result

Created:

- `scripts/audit/high_yield_miss_audit_v12.py`
- `reports/V12_HIGH_YIELD_MISS_AUDIT.md`
- `reports/v12_high_yield_miss_audit.csv`

Key finding:

The worst ML misses are mostly high-yield wheat/barley cases where the model predicts ~2-3 t/ha but target is 4.5-6.4 t/ha.

Example:

- `field_id=209`, `Козлов Увал левый`, wheat_winter 2024;
- target from `productivity_data`: `6.384 t/ha`;
- ML: `2.76..3.10 t/ha`;
- Cropwise: `5.05..5.64 t/ha`;
- operations before as-of: 14-15 operations, 5 fertilizer mix events, 6 seed mix events;
- current v12 feature set has **no explicit seed/fertilizer/chemical operation features**.

Important target-quality note:

For `field_id=209`, 2024, `yield_maps` mean is about `3.31 t/ha`, while `productivity_data` target is `6.384 t/ha`. This row is still suspicious and must be manually verified before using it as a scientific ground truth.

Updated v13 priority:

1. Re-audit conflicting target rows where `productivity_data` and `yield_maps` disagree.
2. Build management features from `operations.csv`:
   - seed mix counts/rates;
   - fertilizer mix counts/rates;
   - chemical mix counts/rates;
   - operation counts before as-of;
   - completed area and timing;
   - sowing operation date from operations, not only history item.
3. Re-train v13 and compare against v12.


## v13 management experiment result

Created:

- `scripts/asof/build_v13_management_features.py`
- `scripts/asof/train_compare_asof_v13.py`
- `scripts/asof/ablate_v13_management.py`
- `reports/asof_v13_results.md`
- `reports/asof_v13_ablation.md`

Full v13 added 57 management columns from `operations.csv`, excluding harvesting operations.
Coverage: ~71% of rows have operations before as-of.

Full v13 result: mostly neutral/slightly worse because the feature set is too wide and sparse for 183 rows.

Ablation result is more useful:

| scenario | date | best compact management variant | MAPE | previous v12 |
|---|---|---|---:|---:|
| all_crops | 1 Jul | nutrients | **28.8%** | 29.3% |
| all_crops | 1 Aug | counts_rates without field_id | **26.5%** | 27.5% |
| wheat_combined | 1 Jul | rates | **27.2%** | 27.4% |
| wheat_combined | 1 Aug | rates | **26.3%** | 28.1% |
| sunflower | 1 Sep | nutrients without field_id | **24.7%** | 25.2% |

Conclusion:

Management data is useful, but only as compact groups. Full operation feature dump overfits/noises the model.

Updated next move:

Build v14 as a curated management model:

- keep only compact management features from ablation:
  - operation counts;
  - fertilizer/seed/chemical rate totals;
  - N/P2O5/K2O/S nutrient proxies;
- drop sparse dates, amounts, max/last columns, custom flags unless proven useful;
- train with no-field-id variants for all-crops/sunflower;
- train with shallow_l30 for wheat rates/nutrients.


## v14 curated management result

Created:

- `scripts/asof/build_v14_curated_management.py`
- `scripts/asof/train_compare_asof_v14.py`
- `reports/asof_v14_results.md`

v14 keeps only 18 compact management features instead of the full 57 from v13.

Result:

| scenario | date | best | MAPE | Cropwise |
|---|---|---|---:|---:|
| all_crops | 1 Jul | v14 | **29.2%** | 29.8% |
| all_crops | 1 Aug | v14 | **26.8%** | 28.9% |
| all_crops | 1 Sep | v12 | **26.9%** | 27.2% |
| sunflower | 1 Jul | v14 | **25.4%** | 31.8% |
| sunflower | 1 Sep | v14 | 24.9% | **17.2%** |
| wheat_combined | all dates | v12 still best | 27.4-31.6% | 29.2-30.9% |

Interpretation:

- Curated management helps global all-crops early/mid-season.
- It does not solve wheat high-yield underprediction.
- v13 ablation showed wheat likes the narrow `rates` subset, not the full curated set.

Next technical move:

Build v15 crop-specific feature policies:

- all_crops: v14 curated management;
- sunflower: v12/v14 depending on as-of date;
- wheat: rates-only management subset or v12 baseline, then inspect high-yield target conflicts.
