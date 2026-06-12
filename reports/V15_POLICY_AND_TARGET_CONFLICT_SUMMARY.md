# v15 policy and target-conflict summary

## What was built

- `scripts/asof/build_v15_rates_management.py`
- `scripts/asof/train_compare_asof_v15_policy.py`
- `scripts/audit/target_conflict_audit_v15.py`
- `reports/asof_v15_policy_results.md`
- `reports/TARGET_CONFLICT_AUDIT_V15.md`
- `reports/target_conflicts_productivity_vs_yield_maps.csv`

## v15 policy result

v15 compares:

- `v12`: corrected target baseline;
- `v14`: curated compact management features;
- `v15_rates`: rates-only management subset.

Best policy by scenario/date:

| scenario | date | best model | ML MAPE | Cropwise MAPE |
|---|---|---|---:|---:|
| all_crops | 1 Jul | v15_rates_shallow_l30 | **29.0%** | 29.8% |
| all_crops | 1 Aug | v14_shallow_l10 | **26.8%** | 28.9% |
| all_crops | 1 Sep | v12_no_field_id | **26.9%** | 27.2% |
| sunflower | 1 Jul | v12_default | **24.0%** | 31.8% |
| sunflower | 1 Aug | v12_no_field_id | 24.6% | **24.5%** |
| sunflower | 1 Sep | v14_shallow_l30 | 24.9% | **17.2%** |
| wheat_combined | 1 Jul | v12_shallow_l30 | **27.4%** | 30.9% |
| wheat_combined | 1 Aug | v15_rates_no_field_id | 27.9% | **27.4%** |
| wheat_combined | 1 Sep | v12_shallow_l10 | 29.7% | **29.2%** |

Interpretation:

- We can beat Cropwise on all-crops at all dates by MAPE, but margin is modest.
- Sunflower is strong early, Cropwise is much stronger late.
- Wheat improves with target correction and rates-only management, but Cropwise remains slightly better after July.
- Feature policy is crop/date-specific; one universal feature set is worse.

## Target conflicts found

`target_conflict_audit_v15.py` compared `productivity_data` fact target vs `yield_maps` combine maps.

Conflicts >1 t/ha:

| field_id | field | year | crop | productivity_data | yield_maps | diff |
|---:|---|---:|---|---:|---:|---:|
| 209 | Козлов Увал левый | 2024 | wheat_winter | 6.384 | 3.310 | +3.074 |
| 195 | Коновалиха 1 | 2023 | wheat_winter | 0.890 | 3.160 | -2.270 |
| 224 | Холодный ключ | 2022 | wheat_spring | 3.079 | 1.753 | +1.326 |
| 230 | Бригадная стрелка | 2022 | sunflower | 3.257 | 1.945 | +1.312 |
| 228 | Бонн 1 | 2022 | wheat_spring | 4.565 | 3.284 | +1.280 |
| 225 | Холодный ключ сенокос | 2021 | sunflower | 1.850 | 0.582 | +1.268 |
| 234 | Красный яр | 2022 | wheat_spring | 3.975 | 2.884 | +1.091 |

Critical point:

`field_id=209`, 2024 is both:

- the worst high-yield ML underprediction;
- a major target conflict: `productivity_data=6.384`, `yield_maps=3.31`.

So the biggest apparent model miss may be a target-quality issue, not a model weakness.

## Next experiment

Build v16 conflict-excluded sensitivity:

- remove rows where `abs(productivity_data - yield_maps) > 1 t/ha`;
- rebuild v12/v14/v15_rates filtered datasets;
- rerun v15 policy;
- check whether wheat/high-yield errors stabilize.

This is necessary before making final scientific claims.
