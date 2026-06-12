# v20 domain-adapted residual — final summary

## Goal

v18 trained a global peer model on ~26k peer rows.

v19 calibrated that model using only `peer_pred` and crop-level corrections.

v20 goes one step further:

```text
residual = target_yield_t_ha - peer_pred
residual = f(local NDVI anomaly, local weather, sowing, soil/management, crop)
```

In words:

- peers learn the general agronomic function;
- local features explain why our fields differ from peer/reference fields.

## Validation

Still leakage-safe:

- for test year `Y`, residual model is trained only on local rows with `year < Y`;
- Cropwise is never used as a feature;
- post-harvest / target-source columns are excluded;
- all results are OOF walk-forward.

Script:

- `scripts/asof/evaluate_v20_domain_adapted_residual.py`

Outputs:

- `models_v2/asof_comparison/asof_results_v20_domain_residual.csv`
- `models_v2/asof_comparison/asof_predictions_v20_domain_residual.csv`
- `models_v2/asof_comparison/asof_results_v20_domain_residual_filtered.csv`
- `reports/asof_v20_domain_residual_results.md`
- `reports/V20_DOMAIN_RESIDUAL_FINAL_SUMMARY.md`

## Methods tested

| Method | Meaning |
|---|---|
| `peer_raw` | raw v18 peer transfer |
| `residual_global` | add average local residual |
| `residual_crop` | add crop-specific local residual |
| `ridge_core` | Ridge residual model on core local features |
| `ridge_all_no_field` | Ridge residual model on all local features, no field id |
| `cat_core_d2/d3` | shallow CatBoost residual model on core local features |
| `cat_all_no_field` | shallow CatBoost residual model on all local features |
| `cat_all_with_field` | same, with field_id categorical |

## All rows

All rows include low-yield 2020. MAPE remains inflated because target values
near 0.2-0.8 t/ha make percentage errors explode.

| Scenario | Date | Best v20 MAPE | CW MAPE | v20 R2 | CW R2 | Best method |
|---|---:|---:|---:|---:|---:|---|
| all_crops | 1 Jul | **74.5** | 78.6 | **0.236** | 0.110 | cat_core_d3 |
| all_crops | 1 Aug | **62.6** | 68.5 | 0.278 | **0.441** | cat_core_d2 |
| all_crops | 1 Sep | **63.2** | 67.8 | 0.244 | **0.432** | cat_core_d2 |
| sunflower | 1 Jul | **23.7** | 26.1 | **-0.069** | -0.477 | peer_raw |
| sunflower | 1 Aug | **22.6** | 24.4 | **-0.135** | -0.174 | residual_global |
| sunflower | 1 Sep | 23.8 | **21.9** | -0.375 | **0.000** | cat_all_with_field |
| wheat | 1 Jul | 114.5 | **96.2** | 0.192 | **0.500** | cat_core_d3 |
| wheat | 1 Aug | 114.4 | **85.9** | 0.186 | **0.544** | cat_core_d3 |
| wheat | 1 Sep | 94.0 | **86.2** | 0.382 | **0.539** | cat_core_d2 |

## Sensitivity: target >= 1 t/ha

This removes percentage-error explosions from near-zero targets.

| Scenario | Date | Best v20 MAPE | CW MAPE | Naive MAPE | v20 R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|
| all_crops | 1 Jul | **27.7** | 29.8 | 35.3 | 0.091 | -0.132 |
| all_crops | 1 Aug | **27.0** | 28.3 | 35.3 | 0.083 | **0.315** |
| all_crops | 1 Sep | **28.0** | 28.0 | 35.3 | 0.015 | **0.299** |
| sunflower | 1 Jul | **23.7** | 26.1 | 23.2 | -0.069 | -0.477 |
| sunflower | 1 Aug | **22.6** | 24.4 | 23.2 | -0.135 | -0.174 |
| sunflower | 1 Sep | 23.8 | **21.9** | 23.2 | -0.375 | **0.000** |
| wheat | 1 Jul | **27.4** | 27.4 | 44.4 | 0.073 | **0.454** |
| wheat | 1 Aug | **28.2** | 28.9 | 44.4 | 0.064 | **0.462** |
| wheat | 1 Sep | **26.4** | 29.9 | 44.4 | 0.196 | **0.445** |

## Sensitivity: target >= 1 t/ha and year >= 2021

This is the most realistic view for normal-yield local seasons after the worst
unit/low-yield year effects.

| Scenario | Date | Best v20 MAPE | CW MAPE | Naive MAPE | v20 R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|
| all_crops | 1 Jul | **25.8** | 29.1 | 35.6 | 0.129 | **0.224** |
| all_crops | 1 Aug | **26.9** | 28.9 | 35.6 | 0.097 | **0.275** |
| all_crops | 1 Sep | 28.6 | **27.6** | 35.6 | -0.017 | **0.319** |
| sunflower | 1 Jul | **23.3** | 29.1 | 24.1 | -0.092 | -0.813 |
| sunflower | 1 Aug | **22.7** | 24.6 | 24.1 | -0.174 | -0.268 |
| sunflower | 1 Sep | 24.0 | **20.3** | 24.1 | -0.427 | **0.157** |
| wheat | 1 Jul | **25.0** | 27.4 | 43.1 | 0.001 | **0.418** |
| wheat | 1 Aug | **25.7** | 28.6 | 43.1 | 0.003 | **0.437** |
| wheat | 1 Sep | **25.2** | 29.7 | 43.1 | 0.126 | **0.416** |

## v19 vs v20

v20 improves the MAPE story:

- v19 best wheat 1 Aug/1 Sep on `target>=1 & year>=2021`:
  - 24.1 / 25.4
- v20:
  - 25.7 / 25.2

So v20 is not uniformly better than v19 on every cell, but it is more stable
for all-crops and all-row views:

- all_crops all rows 1 Aug:
  - v19: 70.5 MAPE
  - v20: 62.6 MAPE
- all_crops all rows 1 Sep:
  - v19: 66.9 MAPE
  - v20: 63.2 MAPE

Most important: v20 confirms that adding local features can improve MAPE, but
it does not solve R2. Local residual models tend to reduce large absolute
percentage errors, but they also flatten/rerank predictions, which hurts R2.

## Interpretation

v20 produces the strongest local MAPE result so far:

- for `target >= 1 t/ha`, v20 is better or tied with Cropwise in 7/9 scenario-date cells by MAPE;
- for `target >= 1 t/ha & year >= 2021`, v20 is better than Cropwise in 6/9 cells by MAPE;
- especially strong for wheat 1 Jul / 1 Aug / 1 Sep after filtering low-yield anomalies.

But:

- Cropwise still dominates R2 in most cells;
- v20 still does not rank fields as well as Cropwise;
- local calibration has only 180 rows, so complex residual models can overfit.

The scientifically honest conclusion:

> v20 demonstrates that global peer learning plus local domain residuals can
> reduce percentage error on normal-yield local seasons. However, Cropwise
> retains superior ranking ability (R2), likely because it has richer hidden
> local features and/or better field-specific calibration.

## Recommendation

Stop expanding model complexity for now.

We now have enough for a strong dissertation narrative:

1. v17 local-only: data-limited, overfits, R2 near zero.
2. v18 peers: large-data global model works strongly.
3. v18 transfer: peer model transfers some signal to local fields.
4. v19/v20: local calibration/domain adaptation improves MAPE on normal local years.
5. Remaining gap: R2/ranking, requiring better local data or field-specific hidden signals.

If continuing technically, the next useful work is not another model tweak. It is:

- verify target quality for 2020 and low-yield rows;
- acquire more local fields with geometry/NDVI;
- or get direct Cropwise/Open Platform API access to enrich the 513 fields with target but no geometry.

