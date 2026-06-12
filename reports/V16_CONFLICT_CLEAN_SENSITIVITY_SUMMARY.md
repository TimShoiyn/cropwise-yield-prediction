# v16 conflict-clean sensitivity summary

## Purpose

v16 tests what happens if we remove rows where two independent target sources disagree strongly:

- `productivity_data` factual yield (`ц/га -> t/ha`);
- `yield_maps` combine-map average yield.

Conflict rule:

`abs(productivity_data_t_ha - yield_map_mean_t_ha) > 1 t/ha`

Removed rows: 7.

## Removed conflict rows

| field_id | field | year | crop | productivity_data | yield_maps | diff |
|---:|---|---:|---|---:|---:|---:|
| 209 | Козлов Увал левый | 2024 | wheat_winter | 6.384 | 3.310 | +3.074 |
| 195 | Коновалиха 1 | 2023 | wheat_winter | 0.890 | 3.160 | -2.270 |
| 224 | Холодный ключ | 2022 | wheat_spring | 3.079 | 1.753 | +1.326 |
| 230 | Бригадная стрелка | 2022 | sunflower | 3.257 | 1.945 | +1.312 |
| 228 | Бонн 1 | 2022 | wheat_spring | 4.565 | 3.284 | +1.280 |
| 225 | Холодный ключ сенокос | 2021 | sunflower | 1.850 | 0.582 | +1.268 |
| 234 | Красный яр | 2022 | wheat_spring | 3.975 | 2.884 | +1.091 |

## v16 result

| scenario | date | best model | ML MAPE | Cropwise MAPE |
|---|---|---|---:|---:|
| all_crops | 1 Jul | v16_v14_shallow_l30 | **27.7%** | 28.7% |
| all_crops | 1 Aug | v16_rates_shallow_l10 | **25.6%** | 28.8% |
| all_crops | 1 Sep | v16_v12_no_field_id | **26.7%** | 27.1% |
| wheat_combined | 1 Jul | v16_v12_no_field_id | **26.6%** | 29.1% |
| wheat_combined | 1 Aug | v16_rates_no_field_id | **25.4%** | 26.8% |
| wheat_combined | 1 Sep | v16_v12_default | **28.3%** | 28.9% |
| sunflower | 1 Jul | v16_v12_default | **26.5%** | 31.4% |
| sunflower | 1 Aug | v16_v12_shallow_l30 | 25.2% | **24.5%** |
| sunflower | 1 Sep | v16_rates_shallow_l10 | 24.4% | **17.3%** |

## Interpretation

This is the strongest evidence so far:

1. Target quality strongly changes the conclusion.
2. With conflict rows removed, ML beats Cropwise by MAPE for:
   - all_crops on all dates;
   - wheat on all dates;
   - sunflower on 1 July.
3. Cropwise remains stronger for sunflower late-season.
4. The worst high-yield ML miss (`field_id=209`, 2024) was also the largest target-source conflict, so it should not be used as unquestioned ground truth.

## Scientific framing

We should present both:

- v15: all corrected productivity targets;
- v16: sensitivity excluding target-source conflicts.

This is honest and stronger scientifically than pretending one target source is perfect.

## Next step

Run v16 error analysis, then consolidate final results table:

- v12 corrected-target baseline;
- v15 policy with management/rates;
- v16 conflict-clean sensitivity;
- Cropwise benchmark.
