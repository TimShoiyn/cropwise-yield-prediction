# No-token path — internal data v10 results

We cannot use Cropwise API, so this sprint uses only local CSVs already in `data_raw`.

## What was built

- `scripts/asof/build_v10_internal_features.py`
  - starts from v9 factual/sowing dataset;
  - adds only honest as-of features from local internal data:
    - `field_scout_reports_aggregated.csv`: report count, max/last growth stage, risk-yield-decreasing flags, field condition, parsed threats;
    - `soil_test_samples.csv` + `soil_tests.csv`: latest/mean pH, P, K, S, N-NO3, organic matter, micronutrients available before forecast date.
- `scripts/asof/train_compare_asof_v10.py`
  - compares v7, v9, v10 and Cropwise with walk-forward validation by year.
- `scripts/asof/ablate_v10_internal_features.py`
  - checks whether scout or soil feature groups help or hurt.
- `scripts/audit/audit_yield_maps_vs_targets.py`
  - compares combine yield maps against factual targets.

## Coverage

v10 still has only 201 rows because NDVI/weather/fields metadata cover only 30 fields.

Internal feature coverage on these rows:

| as-of | scout report coverage | soil sample coverage |
|---|---:|---:|
| 1 Jul | 25.4% | 67.7% |
| 1 Aug | 43.3% | 67.7% |
| 1 Sep | 46.3% | 67.7% |

## Main result

v10 does **not** consistently improve over v9/v7.

Best MAPE from `reports/asof_v10_results.md`:

| scenario | 1 Jul | 1 Aug | 1 Sep | Cropwise pattern |
|---|---:|---:|---:|---|
| sunflower | ML 23.5% | ML 20.9% | ML 23.2% | Cropwise wins only by 1 Sep |
| wheat_combined | ML 36.9% | ML 39.1% | ML 39.0% | Cropwise wins all dates |
| all_crops | ML 34.6% | ML 32.6% | ML 34.7% | mixed, Cropwise slightly stronger overall |

v10 best only clearly appears for `all_crops` on 1 Sep: 34.7% MAPE vs v9 35.8%.
For most cases, v9/v7 are better.

## Ablation conclusion

From `reports/asof_v10_ablation.md`:

- `soil_sample_*` often hurts or adds noise.
- `scout_*` can help late-season (`all_crops` 1 Sep improves when soil is removed), but not consistently.
- Full v10 = too many sparse features for only 201 rows.

This is not a modeling failure. It is data-size + target-quality failure.

## New root causes found

### 1. Dataset size is the hard ceiling

We have:

- factual target available: 2067 rows / 537 fields;
- modelable NDVI/weather/field geometry coverage: 201 rows / 30 fields.

Without API, we cannot expand the main remote-sensing dataset. So the model remains small.

### 2. Target is not fully clean

`yield_maps.csv` gives combine-map yield for 26 fields. Comparing it with factual target found large mismatches:

| field_id | year | crop | target t/ha | yield map t/ha | abs diff |
|---:|---:|---|---:|---:|---:|
| 195 | 2023 | wheat_winter | 0.890 | 3.160 | 2.270 |
| 224 | 2022 | wheat_spring | 3.079 | 1.753 | 1.326 |
| 230 | 2022 | sunflower | 3.257 | 1.945 | 1.312 |
| 228 | 2022 | wheat_spring | 4.565 | 3.284 | 1.280 |
| 225 | 2021 | sunflower | 1.850 | 0.582 | 1.268 |

Rows like this can dominate error because total honest validation is tiny.

### 3. Internal agronomy data is real but sparse

Scout reports contain useful signals (`risk_yield_decreasing`, `field_condition=bad`, BBCH), but coverage before July is only 25%.
So they are useful for discussion/error analysis, not strong enough as generic ML features yet.

## Next no-token steps

1. Build v11 target-clean sensitivity:
   - exclude rows where `yield_maps` and factual target differ by >1 t/ha;
   - retrain v7/v9/v10;
   - check if model accuracy stabilizes.
2. Build `scout_late_season` model only for 1 Sep:
   - use scout features but prune soil;
   - feature set = v9 + scout only;
   - test if late-season scouting genuinely helps.
3. For article/discussion:
   - present API/coverage bottleneck honestly;
   - show that adding more sparse local features does not beat the sample-size ceiling;
   - show target-quality audit with yield maps.


## v11 sensitivity — target-clean with yield maps

After auditing `yield_maps.csv`, we removed 8 `(field_id, year)` rows where combine-map yield and factual target differ by >1 t/ha.

Created:

- `scripts/asof/build_v11_yieldmap_target_clean.py`
- `scripts/asof/train_compare_asof_v11.py`
- `reports/asof_v11_results.md`
- `reports/v11_removed_yieldmap_target_suspects.csv`

### What changed

| scenario | before | after target-clean | interpretation |
|---|---:|---:|---|
| wheat_combined 1 Jul | 36.9-37.3% | **33.0%** | target noise was hurting wheat materially |
| wheat_combined 1 Aug | 39.1% | **37.5%** | better, but Cropwise still ahead |
| wheat_combined 1 Sep | 39.0% | **36.0%** | better, but Cropwise still ahead |
| sunflower 1 Aug | 20.9-24.3% | 22.1% | small/neutral |
| all_crops | mixed | mixed | removing only 8 rows helps wheat more than global model |

### Updated root cause

For the no-token path, the core problem is now clear:

1. We cannot expand from 30 fields to 537 fields without API/geometry, so ML remains underpowered.
2. The available 201-row training set contains several high-impact target mismatches vs yield maps.
3. Sparse internal agronomy features (`scout_*`, `soil_sample_*`) cannot compensate for small N and target noise.
4. Target-cleaning helps wheat, which means the model was partly learning contradictory labels.

### Next best no-token move

Build a wheat-focused model/report using v11 target-clean data, and keep sunflower early-season as the main ML win case:

- sunflower: ML beats Cropwise on 1 Jul / 1 Aug;
- wheat: target-clean narrows the gap, but Cropwise remains stronger;
- discussion: target quality and coverage are limiting factors, not just model choice.
