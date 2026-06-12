"""
Sprint 1.1 — Build clean target_yield_t_ha (in honest tonnes per hectare).

Problem: `productivity` field in Cropwise mixes ц/га and т/га in the same column.
Proof:
  productivity / (harvested_weight / tillable_area) ≈ 10 for cereals & sunflower
  → these are stored in ц/га (centners per hectare).
  But masline crops (rape, soya) sometimes have productivity already in т/га.

Strategy (priority order):
  1) Physical truth: harvested_weight / tillable_area  (т/га, ~129 rows)
  2) productivity from history_items_full.csv → per-crop unit normalization
  3) estimate_value from productivity_estimates.csv → per-crop unit normalization

Output:
  - data_processed/targets_cleaned_t_ha.csv
      (field_id, year, crop_id, target_yield_t_ha, target_source, unit_fix_applied, kept)
  - data_processed/targets_cleaned_audit.csv
      per-(crop, source) statistics: count, % unit-fix, % dropped, dist before/after
  - reports/target_cleaning_audit.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Per-crop realistic max yield in tonnes/ha (used to detect ц/га → т/га).
# If raw value > MAX_T_HA[crop] we assume it's centners/ha and divide by 10.
# Reference: FAO + regional Russia/Kazakhstan production data.
MAX_T_HA: dict[str, float] = {
    "sunflower": 5.0,             # world record ~5.5, regional avg 1.5–2.5
    "wheat_spring": 7.0,
    "wheat_winter": 8.0,
    "barley_spring": 7.0,
    "barley_winter": 8.0,
    "rye_winter": 6.0,
    "avena_spring": 6.0,          # oats
    "maize": 12.0,                # irrigated maize can hit 14, dryland 6–8
    "oil_seed_raps_spring": 4.5,
    "oil_seed_raps_winter": 5.0,
    "soya": 4.0,
    "pea": 4.5,
    "lentil": 3.5,
    "buckwheat": 3.0,
    "safflower": 2.5,
    "medicago": 8.0,              # alfalfa hay (multiple cuts, dry weight)
}
DEFAULT_MAX_T_HA = 5.0

# Per-crop minimum realistic yield (т/га). Below this — drop as bad data.
MIN_T_HA: dict[str, float] = {
    "sunflower": 0.3,
    "wheat_spring": 0.3,
    "wheat_winter": 0.3,
    "barley_spring": 0.3,
    "maize": 0.5,
    "oil_seed_raps_spring": 0.2,
    "oil_seed_raps_winter": 0.2,
    "soya": 0.2,
    "pea": 0.2,
}
DEFAULT_MIN_T_HA = 0.2


def _crop_id_to_std() -> dict[int, str]:
    crops = pd.read_csv(DATA_RAW / "crops.csv")
    out: dict[int, str] = {}
    for _, r in crops[["id", "standard_name"]].dropna(subset=["id"]).iterrows():
        try:
            cid = int(float(r["id"]))
        except Exception:
            continue
        std = str(r.get("standard_name") or "").strip()
        if std:
            out[cid] = std
    return out


def normalize_unit(raw_value: float, std_name: str | None) -> tuple[float, bool]:
    """
    Decide whether raw_value is in ц/га or т/га, return (t_ha_value, was_fixed).
    Heuristic: per-crop max from MAX_T_HA. If raw > max → divide by 10.
    """
    if not pd.notna(raw_value):
        return (np.nan, False)
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if raw_value > cap:
        return (raw_value / 10.0, True)
    return (float(raw_value), False)


def is_in_realistic_range(t_ha: float, std_name: str | None) -> bool:
    if not pd.notna(t_ha):
        return False
    lo = MIN_T_HA.get(std_name or "", DEFAULT_MIN_T_HA)
    hi = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA) * 1.2  # 20% slack post-normalization
    return (t_ha >= lo) and (t_ha <= hi)


def build() -> pd.DataFrame:
    print("=" * 80)
    print("BUILD CLEAN TARGETS (т/га)")
    print("=" * 80)

    crop_id_to_std = _crop_id_to_std()

    # ---------- Source 1: history_items_full.csv (productivity + harvested_weight) ----------
    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False)
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "tillable_area"]].rename(
        columns={"id": "field_id"}
    )

    hi = hi[["field_id", "year", "crop_id", "productivity", "harvested_weight"]].copy()
    hi = hi.merge(fields, on="field_id", how="left")
    hi["std_name"] = hi["crop_id"].map(
        lambda x: crop_id_to_std.get(int(float(x))) if pd.notna(x) else None
    )

    # Source A: physical t/ha = harvested_weight / tillable_area
    hi["physical_t_ha"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(
        hi["tillable_area"], errors="coerce"
    )

    # Source B: productivity normalized
    raw_prod = pd.to_numeric(hi["productivity"], errors="coerce")
    norm_prod_pairs = [normalize_unit(v, s) for v, s in zip(raw_prod, hi["std_name"])]
    hi["productivity_t_ha"] = [p[0] for p in norm_prod_pairs]
    hi["productivity_unit_fixed"] = [p[1] for p in norm_prod_pairs]

    # ---------- Source 2: productivity_estimates.csv (Cropwise own forecast) ----------
    pe = pd.read_csv(DATA_RAW / "productivity_estimates.csv")
    pe = pe[["field_id", "year", "estimate_value"]].copy()

    # We need crop_id for normalization → join from history_items
    crop_per_fy = (
        hi.dropna(subset=["crop_id"])
        .drop_duplicates(subset=["field_id", "year"], keep="first")[["field_id", "year", "crop_id", "std_name"]]
    )
    pe = pe.merge(crop_per_fy, on=["field_id", "year"], how="left")

    raw_est = pd.to_numeric(pe["estimate_value"], errors="coerce")
    norm_est_pairs = [normalize_unit(v, s) for v, s in zip(raw_est, pe["std_name"])]
    pe["estimate_t_ha"] = [p[0] for p in norm_est_pairs]
    pe["estimate_unit_fixed"] = [p[1] for p in norm_est_pairs]

    # ---------- Merge with priority: physical > productivity > estimate ----------
    rows = []

    # Build (field_id, year, crop_id) keys from union of all sources
    keys = pd.concat(
        [
            hi[["field_id", "year", "crop_id", "std_name"]],
            pe[["field_id", "year", "crop_id", "std_name"]],
        ],
        ignore_index=True,
    ).dropna(subset=["field_id", "year"]).drop_duplicates(subset=["field_id", "year"])

    keys["field_id"] = keys["field_id"].astype(int)
    keys["year"] = keys["year"].astype(int)

    hi_lookup = hi.set_index(["field_id", "year"])
    pe_lookup = pe.set_index(["field_id", "year"])

    for _, k in keys.iterrows():
        fid, yr = int(k["field_id"]), int(k["year"])
        std = k["std_name"]
        crop_id = k["crop_id"]
        target = np.nan
        source = None
        fixed = False

        # Priority 1: physical
        if (fid, yr) in hi_lookup.index:
            row = hi_lookup.loc[(fid, yr)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            phys = row.get("physical_t_ha")
            if pd.notna(phys) and is_in_realistic_range(float(phys), std):
                target = float(phys)
                source = "physical_harvested_weight"
                fixed = False

        # Priority 2: productivity (normalized)
        if not pd.notna(target) and (fid, yr) in hi_lookup.index:
            row = hi_lookup.loc[(fid, yr)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prod = row.get("productivity_t_ha")
            if pd.notna(prod) and is_in_realistic_range(float(prod), std):
                target = float(prod)
                source = "productivity_normalized"
                fixed = bool(row.get("productivity_unit_fixed", False))

        # Priority 3: estimate (Cropwise own)
        if not pd.notna(target) and (fid, yr) in pe_lookup.index:
            row = pe_lookup.loc[(fid, yr)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            est = row.get("estimate_t_ha")
            if pd.notna(est) and is_in_realistic_range(float(est), std):
                target = float(est)
                source = "estimate_normalized"
                fixed = bool(row.get("estimate_unit_fixed", False))

        rows.append(
            {
                "field_id": fid,
                "year": yr,
                "crop_id": crop_id if pd.notna(crop_id) else np.nan,
                "std_name": std,
                "target_yield_t_ha": target,
                "target_source": source if source else "no_data",
                "unit_fix_applied": fixed,
                "kept": pd.notna(target),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["field_id", "year"]).reset_index(drop=True)

    out_path = DATA_PROCESSED / "targets_cleaned_t_ha.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"Total rows: {len(out)}, kept: {out['kept'].sum()}, dropped: {(~out['kept']).sum()}")

    return out


def make_audit_report(clean_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("AUDIT REPORT")
    print("=" * 80)

    kept = clean_df[clean_df["kept"]].copy()

    # Per-crop summary
    summary = (
        kept.groupby("std_name")
        .agg(
            n=("target_yield_t_ha", "size"),
            t_min=("target_yield_t_ha", "min"),
            t_p25=("target_yield_t_ha", lambda s: s.quantile(0.25)),
            t_med=("target_yield_t_ha", "median"),
            t_p75=("target_yield_t_ha", lambda s: s.quantile(0.75)),
            t_max=("target_yield_t_ha", "max"),
            n_fixed=("unit_fix_applied", "sum"),
        )
        .sort_values("n", ascending=False)
    )

    # Source breakdown per crop
    src = (
        kept.groupby(["std_name", "target_source"])
        .size()
        .unstack(fill_value=0)
    )

    # Save audit CSV
    audit_csv = DATA_PROCESSED / "targets_cleaned_audit.csv"
    summary_with_src = summary.merge(src, left_index=True, right_index=True, how="left")
    summary_with_src.to_csv(audit_csv)
    print(f"\nSaved {audit_csv}")

    # Markdown report
    md_lines = []
    md_lines.append("# Target Cleaning Audit\n")
    md_lines.append(f"_Generated by `scripts/clean/build_clean_targets.py`_\n\n")
    md_lines.append("## Source priority\n")
    md_lines.append("1. `physical_harvested_weight` — `harvested_weight / tillable_area` (т/га, naturally honest)\n")
    md_lines.append("2. `productivity_normalized` — `history_items_full.productivity` after per-crop unit fix\n")
    md_lines.append("3. `estimate_normalized` — `productivity_estimates.estimate_value` (Cropwise own forecast) after unit fix\n\n")

    md_lines.append("## Summary by crop (kept rows only)\n\n")
    md_lines.append(summary.to_markdown())
    md_lines.append("\n\n## Source distribution by crop\n\n")
    md_lines.append(src.to_markdown())
    md_lines.append("\n\n")

    md_lines.append("## Drop statistics\n\n")
    total_n = len(clean_df)
    kept_n = clean_df["kept"].sum()
    dropped_n = total_n - kept_n
    md_lines.append(f"- Total candidate `(field_id, year)` pairs: **{total_n}**\n")
    md_lines.append(f"- Kept after cleaning: **{kept_n}** ({kept_n / total_n * 100:.1f}%)\n")
    md_lines.append(f"- Dropped (no data, out-of-range, or zero): **{dropped_n}**\n\n")

    md_lines.append("## Unit-fix details\n\n")
    fix_counts = (
        clean_df[clean_df["kept"]]
        .groupby("std_name")["unit_fix_applied"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_fixed", "count": "n_total"})
    )
    fix_counts["pct_fixed"] = (fix_counts["n_fixed"] / fix_counts["n_total"] * 100).round(1)
    md_lines.append(fix_counts.to_markdown())
    md_lines.append("\n\n_If `pct_fixed` is high (>50%), the column was mostly stored in ц/га and we divided by 10._\n")

    report_path = REPORTS / "target_cleaning_audit.md"
    report_path.write_text("".join(md_lines), encoding="utf-8")
    print(f"Saved {report_path}")

    print("\nPer-crop summary:")
    print(summary.to_string())


def main():
    clean_df = build()
    make_audit_report(clean_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
