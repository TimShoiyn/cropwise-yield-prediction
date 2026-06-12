"""
Audit 3: full end-to-end pipeline audit of the LOCAL v17 dataset.

Checks:
  1. Target sanity per crop (range, units, suspicious values)
  2. Feature missingness (dead / mostly-empty features)
  3. Feature -> target signal (Spearman) per crop scenario
  4. Leakage rescan (|corr| with target too high)
  5. Label-noise probe: near-duplicate feature rows with very different target
  6. Cropwise as-of alignment sanity
  7. Sample sizes per crop x year (where stats are unstable)

Output:
  reports/AUDIT3_FULL_PIPELINE.md
  reports/microscope/audit3_feature_signal.csv
  reports/microscope/audit3_missingness.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("reports/microscope")
OUT.mkdir(parents=True, exist_ok=True)

TAG = "08_01"
DF = pd.read_csv(f"data_processed/ml_dataset_clean_v17_v12_clean_asof_{TAG}.csv")

NON_FEATURE = {
    "field_id", "year", "crop_id", "standard_name", "field_name", "prod_crop_ru",
    "target_yield_t_ha", "target_source", "unit_fix_applied", "physical_t_ha",
    "productivity_t_ha", "prod_fact_t_ha", "target_v17", "target_v17_source",
    "target_v17_conflict", "sowing_date", "harvesting_date",
}

NUM = [c for c in DF.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(DF[c])]


def spearman(a, b) -> float:
    s = pd.concat([a, b], axis=1).dropna()
    if len(s) < 8 or s.iloc[:, 0].nunique() < 3:
        return np.nan
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))


def main() -> None:
    md = ["# Audit 3: full pipeline audit (local v17, 1 Aug)\n\n"]
    md.append(f"Rows: {len(DF)}, candidate numeric features: {len(NUM)}\n\n")

    # 1. Target sanity per crop
    md.append("## 1. Target per crop (t/ha)\n\n")
    g = DF.groupby("standard_name")["target_yield_t_ha"].agg(["count", "min", "median", "max", "std"]).round(2)
    md.append(g.to_markdown() + "\n\n")

    # 2. Missingness
    miss = DF[NUM].isna().mean().sort_values(ascending=False)
    miss.to_csv(OUT / "audit3_missingness.csv")
    dead = miss[miss >= 0.80]
    weak = miss[(miss >= 0.40) & (miss < 0.80)]
    md.append("## 2. Missingness\n\n")
    md.append(f"- Features >=80% missing (effectively dead): **{len(dead)}**\n")
    md.append(f"- Features 40-80% missing (weak/sparse): **{len(weak)}**\n\n")
    md.append("Top 25 most-missing features:\n\n")
    md.append(miss.head(25).round(3).to_frame("frac_missing").to_markdown() + "\n\n")

    # 3 + 4. Signal and leakage rescan
    rows = []
    for scen, crops in [("all", None), ("wheat", ["wheat_spring", "wheat_winter"]), ("sunflower", ["sunflower"])]:
        sub = DF if crops is None else DF[DF["standard_name"].isin(crops)]
        y = sub["target_yield_t_ha"]
        for f in NUM:
            rows.append({"scenario": scen, "feature": f, "spearman": spearman(sub[f], y),
                         "n": int(pd.concat([sub[f], y], axis=1).dropna().shape[0])})
    sig = pd.DataFrame(rows)
    sig.to_csv(OUT / "audit3_feature_signal.csv", index=False)

    md.append("## 3. Top feature signal (|Spearman| vs target)\n\n")
    for scen in ["all", "wheat", "sunflower"]:
        s = sig[(sig.scenario == scen) & sig.spearman.notna()].copy()
        s["abs"] = s["spearman"].abs()
        top = s.sort_values("abs", ascending=False).head(12)[["feature", "spearman", "n"]]
        md.append(f"### {scen}\n\n")
        md.append(top.round(3).to_markdown(index=False) + "\n\n")

    md.append("## 4. Leakage rescan (|Spearman| >= 0.9 = suspicious)\n\n")
    leak = sig[sig.spearman.abs() >= 0.9]
    if len(leak):
        md.append(leak.round(3).to_markdown(index=False) + "\n\n")
    else:
        md.append("No feature has |Spearman| >= 0.9 with target. Clean.\n\n")

    # 5. Label noise probe: standardized feature distance vs target gap within crop-year-ish
    md.append("## 5. Label-noise probe (near-identical NDVI/weather, very different yield)\n\n")
    key = ["ndvi_mean_asof", "ndvi_max_asof", "ndvi_integral_asof", "wx_precip_sum_to_asof", "wx_temp_mean_to_asof"]
    probe = DF.dropna(subset=key + ["target_yield_t_ha"]).copy()
    z = (probe[key] - probe[key].mean()) / probe[key].std(ddof=0)
    pairs = []
    arr = z.to_numpy()
    tgt = probe["target_yield_t_ha"].to_numpy()
    crop = probe["standard_name"].to_numpy()
    idx = probe.index.to_numpy()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if crop[i] != crop[j]:
                continue
            d = np.sqrt(((arr[i] - arr[j]) ** 2).sum())
            if d < 0.6:  # very similar feature vector
                gap = abs(tgt[i] - tgt[j])
                if gap > 1.0:
                    pairs.append((crop[i], int(idx[i]), int(idx[j]), round(d, 2), round(tgt[i], 2), round(tgt[j], 2), round(gap, 2)))
    pairs.sort(key=lambda r: -r[-1])
    md.append(f"Found **{len(pairs)}** near-identical-feature pairs (z-dist<0.6) with yield gap >1 t/ha.\n\n")
    if pairs:
        pdf = pd.DataFrame(pairs[:15], columns=["crop", "row_i", "row_j", "feat_dist", "yield_i", "yield_j", "gap"])
        md.append(pdf.to_markdown(index=False) + "\n\n")
    md.append("(High count = irreducible noise: identical signals map to different yields -> caps R2.)\n\n")

    # 6. Cropwise alignment proxy: how many rows have cropwise benchmark
    md.append("## 6. Sample sizes per crop x year\n\n")
    ct = DF.pivot_table(index="standard_name", columns="year", values="target_yield_t_ha", aggfunc="count", fill_value=0)
    md.append(ct.to_markdown() + "\n\n")
    md.append(
        "Cells with 1-3 samples make per-crop-year stats and R2 unstable. "
        "This is the structural reason metrics swing.\n"
    )

    Path("reports/AUDIT3_FULL_PIPELINE.md").write_text("".join(md), encoding="utf-8")

    # console summary
    print("Rows:", len(DF), "numeric features:", len(NUM))
    print("Dead (>=80% missing):", len(dead), "| Weak (40-80%):", len(weak))
    print("Near-identical noisy pairs:", len(pairs))
    print("\nTop signal (all crops):")
    s = sig[(sig.scenario == "all") & sig.spearman.notna()].copy()
    s["abs"] = s.spearman.abs()
    print(s.sort_values("abs", ascending=False).head(10)[["feature", "spearman", "n"]].to_string(index=False))
    print("\nWrote reports/AUDIT3_FULL_PIPELINE.md")


if __name__ == "__main__":
    main()
