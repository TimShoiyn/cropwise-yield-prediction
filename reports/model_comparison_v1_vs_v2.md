# Model comparison: v1 (dirty target, GroupKFold) → v2 (clean target, Walk-Forward CatBoost)

_Generated manually after Sprint 2.1+2.2_

## TL;DR

| | v1 (old GBDT) | v2 (CatBoost, clean) | What changed |
|---|---|---|---|
| Target unit | mixed ц/га + т/га in same column | **honest т/га after per-crop normalization** | unit fix in `build_clean_targets.py` |
| Validation | GroupKFold by year (any year ↔ any year) | **Walk-Forward** (train on past, predict future) | scientifically honest |
| Encoding | crop_id as ordinal int in GBDT | **CatBoost native categorical** for crop_id, prev_crop_id, field_id | GBDT v1 was wrong |
| Trash features | included `field_lat/long`, `field_group_id`, almost-empty soil cols | **dropped** | reduces noise |
| Phase NDVI | included for all crops (mostly NaN) | included only when meaningful for the crop | ok |

## Headline numbers (sunflower)

| Metric | v1 (`models_phases_comparison.csv` baseline) | v2 (CatBoost clean, walk-forward) | Notes |
|---|---|---|---|
| n_rows | 286 | 286 | same data |
| CV R² mean | 0.327 | **0.153 (OOF), -1.69 fold mean** | v1 was inflated: GroupKFold leaks future, target was inconsistent |
| RMSE | 3.84 (in inflated ц/га mixed units!) | **0.58 т/га** | v1 RMSE is meaningless (mixed units) |
| MAPE | 14.9% | **21.4%** | bigger but interpretable: 21% on 1-farm, 286 rows, walk-forward is normal in literature |

> **Why v2 R² looks lower:** because v2 evaluates honestly. v1 metrics on dirty target with a non-walk-forward CV are inflated artifacts — they cannot be reproduced or trusted in publication.

## All scenarios (v2)

```
      scenario  n_total  n_oof    oof_r2  oof_rmse  oof_mae  oof_mape  fold_r2_mean  n_folds
     all_crops      441    252  -0.049    1.090    0.815    35.22%       -0.24          9
     sunflower      286     97   0.153    0.583    0.482    21.38%       -1.69          8
wheat_combined      106     62   0.101    1.084    0.891    43.74%       -1.11          4
```

Per-fold metrics live in:
- `models_v2/catboost_clean_sunflower/fold_metrics.csv`
- `models_v2/catboost_clean_wheat_combined/fold_metrics.csv`
- `models_v2/catboost_clean_all_crops/fold_metrics.csv`

## Diagnostic: why "all_crops" R² is negative?

After unit normalization, target range varies massively across crops:
- maize: 3–6 т/га
- wheat: 2–4 т/га
- sunflower: 1.8–2.6 т/га
- soya/rape: 0.9–2 т/га

A single model is forced to predict near the global mean (~2.4 т/га). Per-crop modeling (sunflower, wheat) is the right direction — and that's exactly what we do.

## Top features (sunflower, CatBoost clean)

```
ndvi_max_season            19.8   ← peak vegetation = peak biomass (canonical)
ndvi_mean_season           10.0
ndvi_early                  9.8
ndvi_mid                    8.7
ndvi_late                   8.6
ndvi_sunflower_p2_mean      3.5   ← phase features actually contribute
ndvi_sunflower_p1_mean      3.2
ndvi_sunflower_p2_max       3.1
prev_crop_id                2.0   ← rotation (sunflower-after-sunflower curse)
field_soil_Mg               1.9   ← soil chemistry
field_soil_pH               1.8
weather_precip_sum_early    1.7   ← drought signal in early season
wx_sunflower_p1_hot_days    1.7
field_id                    1.6   ← inherent field fertility
```

This is **agronomically valid signal**. The model is learning the right things.

## Open issues to address in Sprint 3

1. **OOF scatter shows mean-reversion** (predictions cluster at 2.0–2.7 т/га). Either we need stronger features (DSI/HSI in critical phenology phases) or a less aggressive CatBoost.
2. **Fold R² is highly variable** (std ≈ 3 for sunflower) — models trained on early years (2010–2016) cannot predict modern years (2024–2025) well due to climate shifts. Likely need to drop test_year=2017 from CV (only 12 train years, all very different).
3. **Wheat combined (n=106) is hopeless** — must split into spring vs winter or just train on wheat_spring (n=67). Wheat winter has only 39 rows: report it but do not over-claim.

## Next: Sprint 3.1 — As-of forecast benchmarking against Cropwise

This is where the paper's main contribution lives.
