"""
Follow-up to deep_data_audit.py.

Investigates the most suspicious things found:
  - Field 195: really 18 years of monoculture sunflower?
  - yield_maps.csv: 104 rows but 5.7 MB — what's inside?
  - productivity_data.csv: 3070 rows — yet unused
  - plant_threats.csv: link to fields/years?
  - growth_stages_groups.csv + field_scout_reports: BBCH from agronomist?
  - NDVI cloud_coverage dist (is filter actually applied?)
  - Cropwise estimate accuracy by year (which years they fail)
  - Coverage of v7/v9 (how many factual rows actually trained on?)
  - NDVI per-field annual peak distribution -> are some fields systematically lower?

Output: reports/DATA_DEEP_AUDIT_PART2.md
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures" / "audit2"
FIGS.mkdir(parents=True, exist_ok=True)

OUT: list[str] = []


def line(t: str = "") -> None:
    OUT.append(t + "\n")


def section(t: str) -> None:
    OUT.append(f"\n## {t}\n")


def code(t: str) -> None:
    OUT.append("\n```\n" + t + "\n```\n")


def fig(p: Path, alt: str) -> None:
    OUT.append(f"\n![{alt}]({p.relative_to(REPORTS).as_posix()})\n")


def field_195_history() -> None:
    section("A. Field 195 reality check (18 years of sunflower?)")
    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False,
                     usecols=["field_id", "year", "crop_id", "productivity",
                              "harvested_weight", "sowing_date", "harvesting_date"])
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    hi = hi.merge(crops, on="crop_id", how="left")
    target_field = 195
    g = hi[hi["field_id"] == target_field].sort_values("year")
    line(f"Field {target_field} — n records {len(g)}")
    code(g[["year", "standard_name", "productivity", "harvested_weight",
            "sowing_date", "harvesting_date"]].to_string(index=False))

    line("\nDistinct crops on field 195:")
    code(g["standard_name"].value_counts().to_string())

    # Same for top-N fields by row count
    line("\nTop-10 fields by record count + their crop diversity:")
    def _modal_crop(s: pd.Series) -> str:
        s2 = s.dropna()
        if s2.empty:
            return "?"
        m = s2.mode()
        return str(m.iloc[0]) if len(m) else str(s2.iloc[0])

    def _modal_share(s: pd.Series) -> float:
        s2 = s.dropna()
        if s2.empty:
            return 0.0
        return round(100 * s2.value_counts().iloc[0] / len(s2), 1)

    counts = hi.groupby("field_id").agg(
        n=("year", "size"),
        years=("year", lambda s: f"{int(s.min())}..{int(s.max())}"),
        crops_distinct=("standard_name", "nunique"),
        modal_crop=("standard_name", _modal_crop),
        modal_share_pct=("standard_name", _modal_share),
    ).sort_values("n", ascending=False).head(10)
    code(counts.to_string())


def yield_maps_inspection() -> None:
    section("B. yield_maps.csv inspection")
    ym = pd.read_csv(DATA_RAW / "yield_maps.csv", low_memory=False)
    line(f"rows: {len(ym)}, columns: {ym.shape[1]}")
    code(", ".join(map(str, ym.columns)))
    line("\nField/year distribution:")
    if "field_id" in ym.columns:
        code(ym.groupby("field_id").size().to_string())
    if "created_at" in ym.columns:
        ym["created_at"] = pd.to_datetime(ym["created_at"], errors="coerce")
        line(f"\nDate span of yield_maps: {ym['created_at'].min()} -> {ym['created_at'].max()}")
    if "totals.result.average.value" in ym.columns and "totals.result.average.units" in ym.columns:
        sub = ym[["field_id", "totals.result.average.value", "totals.result.average.units",
                  "calculated_average", "units", "external_average"]].copy()
        line("\nAverage yield from yield_maps (top 30 rows):")
        code(sub.head(30).to_string(index=False))


def productivity_data_inspection() -> None:
    section("C. productivity_data.csv inspection")
    pd_data = pd.read_csv(DATA_RAW / "productivity_data.csv", low_memory=False)
    line(f"rows: {len(pd_data)}, columns: {pd_data.shape[1]}")
    code(", ".join(map(str, pd_data.columns)))
    if pd_data.shape[1] > 0:
        line("\nFirst 10 rows:")
        code(pd_data.head(10).to_string(index=False))
        line("\nDtypes:")
        code(pd_data.dtypes.to_string())


def threats_inspection() -> None:
    section("D. plant_threats + field_scout_reports")
    threats = pd.read_csv(DATA_RAW / "plant_threats.csv", low_memory=False)
    line(f"plant_threats.csv rows: {len(threats)}")
    line("Note: this looks like a static catalog (threats master list), not per-field events.")
    line(f"Distinct threat_type: {threats['threat_type'].value_counts().to_dict()}")

    scout = pd.read_csv(DATA_RAW / "field_scout_reports_aggregated.csv", low_memory=False)
    line(f"\nfield_scout_reports rows: {len(scout)}")
    line("Field/season coverage:")
    if "field_id" in scout.columns and "season" in scout.columns:
        cov = scout.groupby(["field_id", "season"]).size().reset_index(name="n")
        code(cov.head(40).to_string(index=False))

    if "report_time" in scout.columns:
        scout["report_time"] = pd.to_datetime(scout["report_time"], errors="coerce")
        scout["year"] = scout["report_time"].dt.year
        line(f"\nReport time span: {scout['report_time'].min()} -> {scout['report_time'].max()}")
        line(f"Reports per year:")
        code(scout["year"].value_counts().sort_index().to_string())

    if "growth_stage" in scout.columns:
        line("\nTop growth_stage values:")
        code(scout["growth_stage"].value_counts().head(20).to_string())

    if "threats" in scout.columns:
        nonempty = scout["threats"].fillna("").astype(str)
        nonempty_share = (nonempty.str.len() > 5).mean() * 100
        line(f"\nReports with non-empty threats field: {nonempty_share:.1f}%")
        line("Sample threats values:")
        code(nonempty[nonempty.str.len() > 5].head(10).to_string())


def ndvi_cloud_check() -> None:
    section("E. NDVI cloud_coverage actually filtered?")
    ndvi = pd.read_csv(DATA_RAW / "ndvi_timeseries.csv",
                       usecols=["field_id", "year", "date", "ndvi_mean", "cloud_coverage", "data_coverage"])
    line(f"rows: {len(ndvi)}")
    line("\ncloud_coverage distribution:")
    code(ndvi["cloud_coverage"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_string())
    line("\ndata_coverage distribution:")
    code(ndvi["data_coverage"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_string())

    # Per-field per-year median NDVI Jun-Aug
    ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
    season = ndvi[(ndvi["date"].dt.month.isin([6, 7, 8])) & (ndvi["cloud_coverage"].fillna(0) <= 30)]
    by_field = season.groupby("field_id")["ndvi_mean"].agg(["median", "mean", "std"]).round(3)
    line("\nField-level NDVI Jun-Aug median (cloud-clean):")
    code(by_field.sort_values("median").to_string())


def cropwise_year_accuracy() -> None:
    section("F. Cropwise estimate accuracy by year")
    pe = pd.read_csv(DATA_RAW / "productivity_estimates.csv")
    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False,
                     usecols=["id", "field_id", "year", "crop_id", "productivity",
                              "harvested_weight"])
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "tillable_area"]].rename(columns={"id": "field_id"})
    hi = hi.merge(fields, on="field_id", how="left")
    hi["physical_t_ha"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(hi["tillable_area"], errors="coerce")
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    hi = hi.merge(crops, on="crop_id", how="left")

    pe_cols = [c for c in pe.columns if c != "year"]
    pe = pe[pe_cols].merge(
        hi.rename(columns={"id": "history_item_id"})[["history_item_id", "year", "standard_name", "physical_t_ha", "productivity"]],
        on="history_item_id", how="left",
    )

    MAX = {"sunflower": 5.0, "wheat_spring": 7.0, "wheat_winter": 8.0, "barley_spring": 7.0}

    def _fix(v, c):
        if pd.isna(v):
            return np.nan
        cap = MAX.get(c, 5.0)
        return v / 10.0 if v > cap else float(v)

    pe["est_t_ha"] = [_fix(v, c) for v, c in zip(pe["estimate_value"], pe["standard_name"])]
    pe["productivity_norm"] = [_fix(v, c) for v, c in zip(pe["productivity"], pe["standard_name"])]
    pe["fact_t_ha"] = pe["physical_t_ha"].fillna(pe["productivity_norm"])

    cmp = pe.dropna(subset=["est_t_ha", "fact_t_ha"]).copy()
    cmp["err"] = cmp["est_t_ha"] - cmp["fact_t_ha"]
    by_year = cmp.groupby("year").apply(lambda g: pd.Series({
        "n": len(g),
        "bias": g["err"].mean(),
        "mae": g["err"].abs().mean(),
        "rmse": float(np.sqrt((g["err"]**2).mean())),
        "mape_%": float((g["err"].abs() / g["fact_t_ha"]).mean() * 100),
    }), include_groups=False).round(3)
    line("Cropwise estimate accuracy by year:")
    code(by_year.to_string())

    # Per (year, crop)
    by_yc = cmp.groupby(["year", "standard_name"]).apply(lambda g: pd.Series({
        "n": len(g),
        "mae": g["err"].abs().mean(),
        "mape_%": float((g["err"].abs() / g["fact_t_ha"]).mean() * 100),
    }), include_groups=False).round(2)
    line("\nCropwise per (year, crop):")
    code(by_yc.to_string())


def coverage_v7_v9() -> None:
    section("G. v7 / v9 dataset coverage stats")
    for v in ("v7", "v9"):
        for tag in ("07_01", "08_01", "09_01"):
            path = DATA_PROCESSED / f"ml_dataset_clean_{v}_asof_{tag}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            line(f"\n{v} {tag}: rows {len(df)}, fields {df['field_id'].nunique()}, "
                 f"years {int(df['year'].min())}..{int(df['year'].max())}")
            if "standard_name" in df.columns:
                line("Crop breakdown:")
                code(df.groupby("standard_name").size().sort_values(ascending=False).to_string())


def soil_tests_check() -> None:
    section("H. soil_tests (179 numeric features) coverage")
    st = pd.read_csv(DATA_RAW / "soil_tests.csv", low_memory=False)
    line(f"rows: {len(st)}, cols: {st.shape[1]}")
    if "field_id" in st.columns and "made_at" in st.columns:
        line(f"distinct fields with soil tests: {st['field_id'].nunique()}")
        st["made_at"] = pd.to_datetime(st["made_at"], errors="coerce")
        line(f"made_at span: {st['made_at'].min()} -> {st['made_at'].max()}")
        cov = st.groupby("field_id").size().reset_index(name="n_tests")
        code(cov.to_string(index=False))

    samples = pd.read_csv(DATA_RAW / "soil_test_samples.csv", low_memory=False)
    line(f"\nsoil_test_samples rows: {len(samples)}, cols: {samples.shape[1]}")
    nan_pct = samples.isna().mean().sort_values()
    line("Per-column NaN ratio (top 20 most populated):")
    code(nan_pct.head(20).round(3).to_string())


def main() -> None:
    OUT.append("# Deep Data Audit — Part 2 (suspect investigations)\n")
    field_195_history()
    yield_maps_inspection()
    productivity_data_inspection()
    threats_inspection()
    ndvi_cloud_check()
    cropwise_year_accuracy()
    coverage_v7_v9()
    soil_tests_check()
    out = REPORTS / "DATA_DEEP_AUDIT_PART2.md"
    out.write_text("".join(OUT), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
