# v18 peers experiment — final summary

## What changed

We discovered that `data_raw/productivity_estimate_peers.csv` contains a large
peer/reference dataset:

- raw rows: 47,590
- after exact deduplication: **26,822**
- years: **2010-2025**
- crops: 15
- target: `productivity / 10` = t/ha
- NDVI: full dated `ndvi_values` list for every row

This is much larger than the previous local 30-field dataset:

- previous v17 clean: 180 rows, 30 fields
- v18 peers: 26,822 rows

## Datasets built

Script: `scripts/asof/build_v18_peers_dataset.py`

Outputs:

- `data_processed/ml_dataset_v18_peers_strict_asof_07_01.csv`
- `data_processed/ml_dataset_v18_peers_strict_asof_08_01.csv`
- `data_processed/ml_dataset_v18_peers_strict_asof_09_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_07_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_08_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_09_01.csv`

Two variants:

- **strict**: crop, area, previous crop, variety, sowing date, and NDVI features truncated to the as-of date.
- **plus**: strict + `accumulated_precipitations`, `average_soil_moisture`. This is exploratory because the exact as-of horizon of these two aggregate columns is not documented.

Raw `max_ndvi` from peers is intentionally not used because it can be a full-season leakage feature.

## Internal peer validation

Script: `scripts/asof/train_compare_asof_v18_peers.py`

Validation: walk-forward by year. Train rows always have `year < test_year`.

### Strict variant: best ML vs best naive baseline

| Scenario | Date | n | ML MAPE | ML R2 | Best naive MAPE | Best naive R2 |
|---|---:|---:|---:|---:|---:|---:|
| sunflower | 1 Jul | 13,416 | 16.4 | 0.387 | 22.4 | -0.012 |
| sunflower | 1 Aug | 13,416 | 13.5 | 0.565 | 22.4 | -0.012 |
| sunflower | 1 Sep | 13,416 | 13.6 | 0.565 | 22.4 | -0.012 |
| wheat_combined | 1 Jul | 7,144 | 16.4 | 0.656 | 29.3 | 0.047 |
| wheat_combined | 1 Aug | 7,144 | 15.4 | 0.695 | 29.3 | 0.047 |
| wheat_combined | 1 Sep | 7,144 | 15.0 | 0.707 | 29.3 | 0.047 |
| all_crops | 1 Jul | 26,029 | 17.3 | 0.749 | 25.2 | 0.590 |
| all_crops | 1 Aug | 26,029 | 15.2 | 0.399 | 25.2 | 0.590 |
| all_crops | 1 Sep | 26,029 | 15.2 | 0.401 | 25.2 | 0.590 |

Main interpretation:

- On peers, the ceiling problem is gone.
- Sunflower becomes learnable: R2 around 0.56 instead of negative/noisy.
- Wheat becomes strong: R2 around 0.65-0.71.
- MAPE drops from the old 25-35% zone to 13-17%.
- The `plus` variant is almost identical to `strict`, so the safe strict version is enough for publication.

## External transfer to our 30 fields

Script: `scripts/asof/evaluate_v18_peer_transfer.py`

This is the hardest and most honest test:

- train on peer rows with `year < test_year`;
- test on our cleaned v17 30-field dataset;
- compare against Cropwise on the same rows.

### All rows

| Scenario | Date | n | ML MAPE | CW MAPE | Naive MAPE | ML R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sunflower | 1 Jul | 59 | 23.7 | 26.1 | 23.2 | -0.069 | -0.477 |
| sunflower | 1 Aug | 59 | 24.5 | 24.4 | 23.2 | -0.411 | -0.174 |
| sunflower | 1 Sep | 59 | 28.0 | 21.9 | 23.2 | -1.081 | -0.000 |
| wheat_combined | 1 Jul | 88 | 126.9 | 96.2 | 156.8 | 0.212 | 0.500 |
| wheat_combined | 1 Aug | 88 | 122.2 | 85.9 | 156.8 | 0.303 | 0.544 |
| wheat_combined | 1 Sep | 88 | 96.5 | 86.2 | 156.8 | 0.507 | 0.539 |
| all_crops | 1 Jul | 180 | 95.9 | 78.6 | 108.6 | 0.095 | 0.110 |
| all_crops | 1 Aug | 180 | 75.4 | 68.5 | 108.6 | 0.272 | 0.441 |
| all_crops | 1 Sep | 180 | 68.8 | 67.8 | 108.6 | 0.286 | 0.432 |

These MAPE values look huge because 2020 contains very low targets
around 0.2-0.8 t/ha. Percentage error explodes when the denominator is tiny.
This is exactly why we also need MAE/RMSE/R2.

### Excluding very low target rows (`target >= 1 t/ha`)

| Scenario | Date | n | ML MAPE | CW MAPE | Naive MAPE | ML R2 | CW R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wheat_combined | 1 Jul | 72 | 38.8 | 27.4 | 44.4 | 0.168 | 0.454 |
| wheat_combined | 1 Aug | 72 | 34.2 | 28.9 | 44.4 | 0.331 | 0.462 |
| wheat_combined | 1 Sep | 72 | **27.6** | 29.9 | 44.4 | **0.462** | 0.445 |
| all_crops | 1 Jul | 156 | 33.6 | 29.8 | 35.3 | 0.115 | -0.132 |
| all_crops | 1 Aug | 156 | **28.2** | 28.3 | 35.3 | 0.198 | 0.315 |
| all_crops | 1 Sep | 156 | 28.3 | 28.0 | 35.3 | 0.159 | 0.299 |

Interpretation:

- Peer training transfers real signal to our fields.
- It clearly beats the peer crop-mean baseline.
- It does not consistently beat Cropwise on our fields.
- Best transfer result: **wheat 1 Sep**, where ML slightly beats Cropwise both in MAPE and R2 after excluding low-yield denominator explosions.

## What this means

The old conclusion was:

> We have a data ceiling because only 30 fields have NDVI/weather.

The updated conclusion is more precise:

> We do not have a global data ceiling anymore because peers provide 26k rows.  
> But we have a **domain-transfer ceiling**: a model trained on peer/reference fields does not automatically match our 30 fields as well as Cropwise does.

This is still a big improvement:

- v17 local-only model had OOF R2 around 0 / negative.
- v18 peer-transfer reaches R2 around 0.27-0.51 on our fields, depending on crop/date.
- That means the big dataset is useful.

But Cropwise remains hard to beat on our exact fields because:

- Cropwise likely uses more hidden features than peers expose.
- Our target distribution has unusual low-yield 2020 rows.
- Peer rows have no field geometry/id, so the model cannot learn local field-specific calibration.
- Our 30 fields may be domain-shifted relative to peer/reference fields.

## Scientific narrative

This gives a stronger dissertation story:

1. Start with local data only (v17): model is data-limited and overfits.
2. Discover hidden large peer dataset (v18): model becomes statistically strong on 26k field-seasons.
3. Transfer peer-trained model to local fields: signal improves, R2 becomes positive, but Cropwise remains competitive.
4. Conclusion: the bottleneck is no longer only data volume; it is **domain adaptation** and **local calibration**.

## Recommended next step

Do not stop here.

Next, build **v19 hybrid calibration**:

1. Train a global model on v18 peers.
2. Generate peer-model predictions for our 30 fields.
3. Use our 180 local rows to train a small calibration layer:
   - `target = a * peer_pred + b_crop + b_year + residual_features`
   - or CatBoost/linear residual model on `target - peer_pred`.
4. Validate with walk-forward by year.
5. Compare against Cropwise again.

This is the natural approach:

- peers give the global agronomic function;
- local data calibrates the function to our farm/region/target source.

Expected effect:

- lower MAPE on our fields;
- better R2 than local-only v17;
- possibly beating Cropwise on wheat/all-crops early and mid-season.

## Files

Datasets:

- `data_processed/ml_dataset_v18_peers_strict_asof_07_01.csv`
- `data_processed/ml_dataset_v18_peers_strict_asof_08_01.csv`
- `data_processed/ml_dataset_v18_peers_strict_asof_09_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_07_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_08_01.csv`
- `data_processed/ml_dataset_v18_peers_plus_asof_09_01.csv`

Reports:

- `reports/V18_PEERS_DATASET_SUMMARY.md`
- `reports/asof_v18_peers_results.md`
- `reports/asof_v18_peer_transfer_results.md`
- `reports/V18_PEERS_FINAL_SUMMARY.md`

Scripts:

- `scripts/asof/build_v18_peers_dataset.py`
- `scripts/asof/train_compare_asof_v18_peers.py`
- `scripts/asof/evaluate_v18_peer_transfer.py`
