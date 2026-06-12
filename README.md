# Cropwise Yield Prediction (NDVI + Weather + Soil)

This repository contains a reproducible pipeline to:

- export agronomic data from **Cropwise Operations API** (GET only),
- build ML-ready datasets at the **field × year** level,
- train and evaluate yield prediction models with **honest validation** (GroupKFold by year, OOF predictions),
- compare against **Cropwise predictions**, including **true time-series forecasts** from `estimate_history` (as-of dates).

The project was built for research (NIRD / supervisor review) with strong focus on:
- avoiding leakage (all preprocessing inside sklearn Pipeline),
- keeping artifacts on disk (CSV/PNG) for transparent reporting.

## Quick start

### 1) Create environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Set API token (do not hardcode in code)

```powershell
$env:CROPWISE_API_KEY = "YOUR_TOKEN_HERE"
```

Never commit tokens to GitHub. If a token was posted/shared, revoke and re-issue it.

### 3) Export raw data (Operations API)

```powershell
python fetch_cropwise_data.py
```

Optional (Cropwise yield extras):

```powershell
python fetch_productivity_extras.py
```

### 4) Build ML datasets

```powershell
python build_ml_dataset.py
```

### 5) Train baseline models (OOF + GroupKFold by year)

```powershell
python train_baseline_model.py
```

Outputs go to `models/` (OOF predictions, plots, `models_comparison.csv`).

---

## Repository structure

```
.
├── data_raw/                           # Raw exports (CSV)
│   ├── fields.csv
│   ├── crops.csv
│   ├── operations.csv
│   ├── productivity_estimates.csv
│   ├── ndvi_timeseries.csv
│   ├── weather_history_items.csv
│   ├── soil_tests.csv
│   ├── productivity_estimate_histories.csv    # Cropwise forecast history (estimate_history)
│   └── productivity_estimate_peers.csv
├── data_processed/                     # ML-ready datasets (CSV)
│   ├── ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv
│   ├── ml_dataset_full_extended_cropwindow_v1_phases.csv
│   └── ...
├── models/                             # Model artifacts (CSV/PNG/PKL)
│   ├── oof_*.csv
│   ├── models_comparison.csv
│   ├── models_phases_comparison.csv
│   ├── cropwise_asof_metrics.csv
│   └── plots/
├── reports/                            # Reports and slide-ready summaries
│   └── SLIDES_DATA_FINAL_PHASES_AND_CROPWISE_ASOF.md
├── build_ml_dataset.py
├── train_baseline_model.py
└── requirements.txt
```

## Datasets (what is predicted)

Core supervised learning table is **field × year** with:

- **Target**: `target_yield_t_ha` (yield in t/ha)
- **Features**: NDVI seasonal aggregates, crop_id/prev_crop_id, geometry, soil tests, daily-weather aggregates (crop-window), etc.

Main crop-window dataset:
- `data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv`

## Key evaluation principle (no leakage)

- Cross-validation is **GroupKFold by year** (leave-year-out style).
- Imputation is done **inside** sklearn `Pipeline` during CV.
- Reported metrics use **OOF predictions** (not in-sample train metrics).

## Experiments included

### A) Phase-based features (GDD phases)

Script:
- `train_phases_experiment.py`

Outputs:
- dataset with phase features: `data_processed/ml_dataset_full_extended_cropwindow_v1_phases.csv`
- metrics: `models/models_phases_comparison.csv`
- plots and short summary: run `python make_phases_artifacts.py`

Key plots:
- `models/plots/phases_oof_scatter_allcrops_baseline_vs_phases.png`
- `models/plots/phases_cv_r2_barplot_3subsets.png`
- `models/plots/phases_feature_importance_allcrops_phases_top20.png`

### B) Cropwise as-of forecasts (true time-series from API)

The correct “honest” Cropwise forecast source is:
- `data_raw/productivity_estimate_histories.csv` (`estimate_history` = date → forecast)

Scripts:
- `cropwise_asof_eval.py` → creates:
  - `models/cropwise_asof_joined.csv`
  - `models/cropwise_asof_metrics.csv`
- `cropwise_asof_vs_model.py` → creates:
  - `models/cropwise_asof_vs_model_joined.csv`
  - `models/cropwise_asof_vs_model_metrics.csv`

Important note:
- `final` forecast often equals actual yield (post-season update), so **as-of dates (07/01, 08/01, 09/01)** are used for honest comparison.

## Debug / sanity checks

```powershell
python debug_dataset_sanity.py
```

Artifacts go to `reports/debug/`.

## Slides / supervisor-ready summary

Use:
- `reports/SLIDES_DATA_FINAL_PHASES_AND_CROPWISE_ASOF.md`

It contains:
- key numbers,
- paths to all CSV/PNG artifacts,
- a slide skeleton (2–3 slides).

## Notes on NDVI access

NDVI export depends on Open Platform / Remote Sensing access. If `fetch_ndvi_timeseries.py` returns 401/403, you need:
- Remote Sensing contract enabled for your account, or
- a separate token for the Open Platform API.

## License

Internal research project. Add license if you plan public reuse.

## Additional docs

See `README_ML.md` for a more detailed, step-by-step ML pipeline description.

