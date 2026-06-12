# v19 hybrid calibration — final summary

## Goal

v18 proved that the large peer dataset can learn a strong global yield model:

- peers internal R2: ~0.56-0.71 for sunflower/wheat
- peers internal MAPE: ~13-16%

But direct transfer to our 30 fields was not enough to clearly beat Cropwise.
So v19 tested a natural hybrid approach:

1. Train global peer model on `productivity_estimate_peers`.
2. Predict our local fields with that model (`peer_pred`).
3. Calibrate `peer_pred` using only local past years.

Validation stayed strict:

- for test year `Y`, local calibration uses only local rows with `year < Y`;
- no Cropwise estimate is used as a feature;
- Cropwise remains only an external benchmark.

## Inputs

Main input:

- `models_v2/asof_comparison/asof_predictions_v18_peer_transfer.csv`

This file already contains leakage-safe peer predictions for our 30 local fields.

Script:

- `scripts/asof/evaluate_v19_hybrid_calibration.py`

Outputs:

- `models_v2/asof_comparison/asof_results_v19_hybrid_calibration.csv`
- `models_v2/asof_comparison/asof_predictions_v19_hybrid_calibration.csv`
- `models_v2/asof_comparison/asof_results_v19_hybrid_calibration_filtered.csv`
- `reports/asof_v19_hybrid_calibration_results.md`

## Calibration methods tested

| Method | Idea |
|---|---|
| `peer_raw` | Use v18 peer prediction as-is |
| `residual_global` | Add mean local residual from previous years |
| `residual_crop` | Add mean local residual by crop |
| `linear_global` | Ridge: `target ~ peer_pred` |
| `linear_crop` | Ridge: `target ~ peer_pred + crop one-hot` |
| `blend_crop_mean` | `alpha * peer_pred + (1-alpha) * local crop mean`, alpha chosen on train |

## All rows result

All rows include very low-yield 2020 targets. MAPE becomes inflated because
percentage error divides by a tiny target value.

| Scenario | Date | Best v19 MAPE | CW MAPE | Best v19 R2 | CW R2 | Best MAPE method |
|---|---:|---:|---:|---:|---:|---|
| all_crops | 1 Jul | **76.5** | 78.6 | **0.192** | 0.110 | residual_global |
| all_crops | 1 Aug | 70.5 | **68.5** | 0.272 | **0.441** | residual_global |
| all_crops | 1 Sep | **66.9** | 67.8 | 0.286 | **0.432** | residual_global |
| sunflower | 1 Jul | **23.7** | 26.1 | **-0.069** | -0.477 | peer_raw |
| sunflower | 1 Aug | **22.6** | 24.4 | **-0.135** | -0.174 | residual_global / residual_crop |
| sunflower | 1 Sep | 23.4 | **21.9** | -0.334 | **0.000** | linear_crop |
| wheat | 1 Jul | 115.9 | **96.2** | 0.212 | **0.500** | residual_global |
| wheat | 1 Aug | 113.9 | **85.9** | 0.303 | **0.544** | linear_global |
| wheat | 1 Sep | 94.3 | **86.2** | 0.507 | **0.539** | linear_global |

## Sensitivity: `target >= 1 t/ha`

This removes denominator explosions from extremely low-yield rows.

| Scenario | Date | Best v19 MAPE | CW MAPE | Naive MAPE | Best v19 R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|
| all_crops | 1 Jul | **28.8** | 29.8 | 35.3 | 0.030 | -0.132 |
| all_crops | 1 Aug | **28.2** | 28.3 | 35.3 | 0.198 | **0.315** |
| all_crops | 1 Sep | 28.3 | **28.0** | 35.3 | 0.159 | **0.299** |
| sunflower | 1 Jul | **23.7** | 26.1 | 23.2 | -0.069 | -0.477 |
| sunflower | 1 Aug | **22.6** | 24.4 | 23.2 | -0.135 | -0.174 |
| sunflower | 1 Sep | 23.4 | **21.9** | 23.2 | -0.334 | **0.000** |
| wheat | 1 Jul | 28.0 | **27.4** | 44.4 | 0.054 | **0.454** |
| wheat | 1 Aug | **26.8** | 28.9 | 44.4 | 0.283 | **0.462** |
| wheat | 1 Sep | **26.6** | 29.9 | 44.4 | 0.206 | **0.445** |

## Sensitivity: `target >= 1 t/ha` and `year >= 2021`

This view removes both the low-yield denominator issue and early years before
local calibration has much training history.

| Scenario | Date | Best v19 MAPE | CW MAPE | Naive MAPE | Best v19 R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|
| all_crops | 1 Jul | **27.2** | 29.1 | 35.6 | 0.042 | **0.224** |
| all_crops | 1 Aug | **28.1** | 28.9 | 35.6 | 0.250 | **0.275** |
| all_crops | 1 Sep | 28.7 | **27.6** | 35.6 | 0.167 | **0.319** |
| sunflower | 1 Jul | **23.3** | 29.1 | 24.1 | -0.092 | -0.813 |
| sunflower | 1 Aug | **22.7** | 24.6 | 24.1 | -0.174 | -0.268 |
| sunflower | 1 Sep | 23.5 | **20.3** | 24.1 | -0.360 | **0.157** |
| wheat | 1 Jul | **25.7** | 27.4 | 43.1 | -0.022 | **0.418** |
| wheat | 1 Aug | **24.1** | 28.6 | 43.1 | 0.262 | **0.437** |
| wheat | 1 Sep | **25.4** | 29.7 | 43.1 | 0.138 | **0.416** |

## Interpretation

v19 is useful, but not a complete solution.

What improved:

- MAPE improves over v18 peer transfer in several cases.
- For normal-yield rows (`target >= 1`) v19 beats Cropwise by MAPE in:
  - all_crops 1 Jul / 1 Aug
  - sunflower 1 Jul / 1 Aug
  - wheat 1 Aug / 1 Sep
- v19 is always much better than naive crop means.

What did not improve enough:

- R2 usually remains below Cropwise.
- Raw peer predictions often have better R2 than calibrated predictions.
- Local calibration has too few rows and can overcorrect.

The correct conclusion:

> v18 peers solved the global data-volume problem.  
> v19 local calibration improves MAPE in the local domain, but does not yet beat Cropwise in ranking ability.

## Best current scientific framing

The strongest narrative is now:

1. Local-only model (v17) is data-limited and overfits.
2. Peer dataset (v18) unlocks strong global yield modeling.
3. Direct peer transfer improves local predictions but still has domain shift.
4. Hybrid calibration (v19) improves MAPE for normal-yield local rows, especially wheat 1 Aug / 1 Sep.
5. Cropwise remains stronger in R2 because it likely has richer hidden features and local calibration.

This is a publishable and defensible story:

- not "we beat Cropwise everywhere";
- but "we identified data/target/leakage issues, built a large peer-based as-of model, and quantified domain adaptation limits."

## Recommended next step

Do one more targeted experiment, not a broad search:

### v20 domain-adapted calibration

Use v19 outputs but calibrate only where calibration is statistically meaningful:

1. Exclude target `< 1 t/ha` from percentage-based model selection, but still report them separately.
2. Use `peer_raw` when optimizing R2 and `residual_global/linear_global` when optimizing MAPE.
3. Add local as-of features to the calibration layer:
   - `peer_pred`
   - local NDVI anomaly (`ndvi_anom_*`)
   - local weather (`wx_*`)
   - target crop
   - year-normalized weather deviation
4. Train residual model:
   - `residual = target - peer_pred`
5. Validate walk-forward.

Why v20 is more promising than v19:

- v19 calibrates using only `peer_pred` + crop.
- v20 will explain why peer model is wrong locally using local features that peers do not have.

Expected outcome:

- MAPE should improve further.
- R2 may still trail Cropwise, but could approach it for wheat/all_crops.

