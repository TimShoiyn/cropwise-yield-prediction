"""
Audit 3b: three cheap, high-value analyses on the local v17 dataset.

A. Noise ceiling (kNN delta / Gamma test): estimates the irreducible noise
   variance, hence the MAX achievable R2 with the current features.
B. Feature pruning: full (148) vs core (~22) features, walk-forward CatBoost,
   compare pooled R2 / within-crop-year R2 / MAPE and stability.
C. Leave-one-field-out (LOFO) CV: tests spatial generalization to unseen fields.

Outputs:
  reports/AUDIT3B_CHEAP_WINS.md
  reports/microscope/audit3b_*.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("reports/microscope")
OUT.mkdir(parents=True, exist_ok=True)

TAG = "08_01"
DF = pd.read_csv(f"data_processed/ml_dataset_clean_v17_v12_clean_asof_{TAG}.csv")
DF = DF.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)

CORE = [
    "ndvi_mean_asof", "ndvi_max_asof", "ndvi_p75_asof", "ndvi_integral_asof",
    "ndvi_amplitude_asof", "ndvi_anom_mean_asof", "ndvi_above_05_days_asof",
    "ndvi_slope_asof", "ndvi_std_asof",
    "wx_precip_sum_to_asof", "wx_temp_mean_to_asof", "wx_vpd_max_to_asof",
    "wx_water_balance_to_asof", "wx_srad_sum_to_asof", "wx_may_precip",
    "wx_jul_vpd_max", "wx_may_srad",
    "days_after_sowing_asof", "sowing_doy",
    "years_since_wheat", "years_since_sunflower",
    "field_tillable_area", "field_soil_pH", "field_soil_OM",
]
NON_FEATURE = {
    "field_id", "year", "standard_name", "field_name", "prod_crop_ru",
    "target_yield_t_ha", "target_source", "unit_fix_applied", "physical_t_ha",
    "productivity_t_ha", "prod_fact_t_ha", "target_v17", "target_v17_source",
    "target_v17_conflict", "sowing_date", "harvesting_date",
}
FULL = [c for c in DF.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(DF[c])]

SCENARIOS = [("all", None), ("wheat", ["wheat_spring", "wheat_winter"]), ("sunflower", ["sunflower"])]
CB = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=20.0,
          loss_function="RMSE", random_seed=42, verbose=False, allow_writing_files=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 3:
        return dict(r2=np.nan, mae=np.nan, mape=np.nan)
    return dict(
        r2=float(r2_score(y, p)),
        mae=float(mean_absolute_error(y, p)),
        mape=float(np.mean(np.abs((y - p) / np.where(y > 0, y, np.nan))) * 100),
    )


def wcy_r2(df, ycol, pcol):
    s = df.dropna(subset=[ycol, pcol, "standard_name", "year"])
    if len(s) < 4:
        return np.nan
    y, p = s[ycol].astype(float), s[pcol].astype(float)
    g = [s["standard_name"], s["year"]]
    yd = y - y.groupby(g).transform("mean")
    pd_ = p - p.groupby(g).transform("mean")
    tot = float((yd ** 2).sum())
    return np.nan if tot <= 1e-9 else 1 - float(((yd - pd_) ** 2).sum()) / tot


# ---------- A. Noise ceiling (kNN delta test) ----------
def noise_ceiling(df, feats, group="standard_name"):
    rows = []
    for crop, g in df.groupby(group):
        sub = g.dropna(subset=feats + ["target_yield_t_ha"])
        if len(sub) < 8:
            continue
        X = sub[feats].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = sub["target_yield_t_ha"].to_numpy(float)
        # nearest neighbor delta test
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = d2.argmin(1)
        delta = 0.5 * np.mean((y - y[nn]) ** 2)  # noise variance estimate
        vy = np.var(y)
        rows.append(dict(crop=crop, n=len(sub), var_y=round(vy, 3),
                         noise_var=round(delta, 3),
                         max_r2=round(1 - delta / vy, 3) if vy > 1e-9 else np.nan))
    return pd.DataFrame(rows)


# ---------- B. Feature pruning (walk-forward) ----------
def walk_forward(df, feats, crops):
    sub = df if crops is None else df[df["standard_name"].isin(crops)]
    sub = sub.reset_index(drop=True)
    use = [f for f in feats if f in sub.columns]
    pred = np.full(len(sub), np.nan)
    for ty in sorted(sub["year"].unique()):
        tr = sub[sub["year"] < ty]
        te_mask = (sub["year"] == ty).to_numpy()
        if len(tr) < 20:
            continue
        Xtr = tr[use].apply(pd.to_numeric, errors="coerce")
        Xte = sub.loc[te_mask, use].apply(pd.to_numeric, errors="coerce")
        m = CatBoostRegressor(**CB)
        m.fit(Pool(Xtr, tr["target_yield_t_ha"].astype(float)))
        pred[te_mask] = m.predict(Xte)
    sub = sub.assign(pred=pred)
    cmp = sub.dropna(subset=["pred"])
    return cmp, metrics(cmp["target_yield_t_ha"], cmp["pred"]), wcy_r2(cmp, "target_yield_t_ha", "pred")


# ---------- C. Leave-one-field-out ----------
def lofo(df, feats, crops):
    sub = df if crops is None else df[df["standard_name"].isin(crops)]
    sub = sub.reset_index(drop=True)
    use = [f for f in feats if f in sub.columns]
    pred = np.full(len(sub), np.nan)
    for fid in sub["field_id"].unique():
        tr = sub[sub["field_id"] != fid]
        te_mask = (sub["field_id"] == fid).to_numpy()
        if len(tr) < 20:
            continue
        Xtr = tr[use].apply(pd.to_numeric, errors="coerce")
        Xte = sub.loc[te_mask, use].apply(pd.to_numeric, errors="coerce")
        m = CatBoostRegressor(**CB)
        m.fit(Pool(Xtr, tr["target_yield_t_ha"].astype(float)))
        pred[te_mask] = m.predict(Xte)
    cmp = sub.assign(pred=pred).dropna(subset=["pred"])
    return metrics(cmp["target_yield_t_ha"], cmp["pred"]), wcy_r2(cmp, "target_yield_t_ha", "pred")


def main():
    md = ["# Audit 3b: cheap wins (noise ceiling, pruning, LOFO)\n\n"]

    # A
    nc = noise_ceiling(DF, CORE)
    nc.to_csv(OUT / "audit3b_noise_ceiling.csv", index=False)
    md.append("## A. Noise ceiling (max achievable R2 per crop)\n\n")
    md.append("kNN delta test: noise_var = 0.5*mean((y - y_nearest)^2); max_r2 = 1 - noise_var/var_y.\n\n")
    md.append(nc.to_markdown(index=False) + "\n\n")
    md.append("Interpretation: even a perfect model cannot exceed `max_r2` with current features.\n\n")
    print("=== A. Noise ceiling ===")
    print(nc.to_string(index=False))

    # B
    md.append("## B. Feature pruning: full vs core (walk-forward)\n\n")
    rowsB = []
    for name, crops in SCENARIOS:
        _, mf, wf = walk_forward(DF, FULL, crops)
        _, mc, wc = walk_forward(DF, CORE, crops)
        rowsB.append(dict(scenario=name, set="full", n_feat=len(FULL),
                          r2=round(mf["r2"], 3), wcy_r2=round(wf, 3) if wf == wf else np.nan,
                          mape=round(mf["mape"], 1)))
        rowsB.append(dict(scenario=name, set="core", n_feat=len(CORE),
                          r2=round(mc["r2"], 3), wcy_r2=round(wc, 3) if wc == wc else np.nan,
                          mape=round(mc["mape"], 1)))
    dfB = pd.DataFrame(rowsB)
    dfB.to_csv(OUT / "audit3b_pruning.csv", index=False)
    md.append(dfB.to_markdown(index=False) + "\n\n")
    print("\n=== B. Pruning full vs core ===")
    print(dfB.to_string(index=False))

    # C
    md.append("## C. Leave-one-field-out CV (spatial generalization, core features)\n\n")
    rowsC = []
    for name, crops in SCENARIOS:
        mc, wc = lofo(DF, CORE, crops)
        rowsC.append(dict(scenario=name, lofo_r2=round(mc["r2"], 3),
                          lofo_wcy_r2=round(wc, 3) if wc == wc else np.nan,
                          lofo_mape=round(mc["mape"], 1)))
    dfC = pd.DataFrame(rowsC)
    dfC.to_csv(OUT / "audit3b_lofo.csv", index=False)
    md.append(dfC.to_markdown(index=False) + "\n\n")
    md.append(
        "LOFO trains on all fields except one and predicts the held-out field. "
        "It answers: can the model predict a field it has never seen? Low/negative "
        "R2 here = weak spatial transfer, consistent with the non-persistent field effect.\n"
    )
    print("\n=== C. LOFO ===")
    print(dfC.to_string(index=False))

    Path("reports/AUDIT3B_CHEAP_WINS.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote reports/AUDIT3B_CHEAP_WINS.md")


if __name__ == "__main__":
    main()
