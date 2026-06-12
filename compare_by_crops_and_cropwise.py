"""
Сравнение качества модели по культурам и с прогнозами Cropwise.

Задачи:
1. Собрать общий датасет: (field_id, year, crop_id, target_yield_t_ha, model_pred_t_ha, cropwise_pred_t_ha).
2. Группировка по культурам и crop_group (зерновые, масличные, картофель, свёкла и т.д.).
3. Метрики R², RMSE, MAE, MAPE по культурам/группам для нашей модели и для Cropwise.
4. Таблица сравнения и черновой текст для раздела 5 статьи.

Входные файлы:
- data_raw/productivity_data.csv — колонки: Поле, Год, Культура, урожайность факт ц/га, урожайность прогноз ц/га.
- data_raw/fields.csv — id, name (для маппинга Поле -> field_id).
- data_raw/crops.csv — id, name (для маппинга Культура -> crop_id и групп).
- data_processed/ml_dataset_full_extended.csv, models/oof_predictions_full_extended_all_no_leak.csv (предпочтительно).

Если имена полей в productivity_data.Поле не совпадают с fields.name, строк с Cropwise будет 0;
метрики по культурам для нашей модели всё равно считаются.

Запуск: из корня проекта выполнить: python compare_by_crops_and_cropwise.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Корень проекта
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

# Пути
DATA_RAW = ROOT_DIR / "data_raw"
DATA_PROCESSED = ROOT_DIR / "data_processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

FIELDS_CSV = DATA_RAW / "fields.csv"
CROPS_CSV = DATA_RAW / "crops.csv"
PRODUCTIVITY_DATA_CSV = DATA_RAW / "productivity_data.csv"
ML_FULL_EXTENDED_CSV = DATA_PROCESSED / "ml_dataset_full_extended.csv"
# Предпочтительно используем OOF-предсказания от лучшей модели (Iteration 2, регуляризованный GBDT).
OOF_PRED_REG_CSV = MODELS_DIR / "oof_predictions_full_extended_all_no_leak_reg.csv"
OOF_PRED_BASE_CSV = MODELS_DIR / "oof_predictions_full_extended_all_no_leak.csv"

# Выходные файлы
OUT_COMPARISON_CSV = DATA_PROCESSED / "comparison_model_vs_cropwise.csv"
OUT_METRICS_BY_CROP_CSV = TABLES_DIR / "metrics_by_crop_model_vs_cropwise.csv"
OUT_METRICS_BY_GROUP_CSV = TABLES_DIR / "metrics_by_crop_group_model_vs_cropwise.csv"
OUT_DRAFT_TEXT_MD = TABLES_DIR / "draft_section_5_crops_comparison.md"


# --- Группы культур (crop_id -> crop_group) по справочнику и типу ---
def build_crop_group_mapping(crops_df: pd.DataFrame):
    """Возвращает функцию crop_id -> crop_group (grain, oilseeds, potato, sugar_beet, other)."""
    grain_ids = {
        1, 2, 3, 4, 5, 6, 7, 13, 16, 18, 23, 24, 30, 31, 32, 35, 36, 41, 42, 43, 44,
    }  # овёс, ячмень, гречиха, горох, нут, чечевица, кукуруза, просо, тритикале, пшеница, рожь, соя
    oilseed_ids = {15, 20, 21, 33, 40}  # лён, рапс яровой/озимый, сафлор, подсолнечник
    potato_ids = {29}
    sugar_beet_ids = {38}

    def group_of(cid):
        if pd.isna(cid):
            return "other"
        cid = int(float(cid))
        if cid in grain_ids:
            return "grain"
        if cid in oilseed_ids:
            return "oilseeds"
        if cid in potato_ids:
            return "potato"
        if cid in sugar_beet_ids:
            return "sugar_beet"
        return "other"

    return group_of


def load_field_name_to_id() -> dict:
    """Маппинг name -> id из fields.csv (для связи productivity_data.Поле с field_id)."""
    if not FIELDS_CSV.exists():
        return {}
    df = pd.read_csv(FIELDS_CSV, usecols=["id", "name"], encoding="utf-8")
    df["name"] = df["name"].astype(str).str.strip()
    return df.set_index("name")["id"].to_dict()


def load_crop_name_to_id() -> dict:
    """Маппинг название культуры (рус.) -> crop id из crops.csv."""
    if not CROPS_CSV.exists():
        return {}
    df = pd.read_csv(CROPS_CSV, usecols=["id", "name"], encoding="utf-8")
    df["name"] = df["name"].astype(str).str.strip()
    return df.set_index("name")["id"].to_dict()


def prepare_cropwise_dataset() -> pd.DataFrame:
    """
    Читает productivity_data.csv, переводит урожайность в т/га,
    добавляет field_id и crop_id. Одна строка на (field_id, year).
    """
    if not PRODUCTIVITY_DATA_CSV.exists():
        print(f"[WARN] {PRODUCTIVITY_DATA_CSV} не найден. Пропуск Cropwise.")
        return pd.DataFrame()

    for enc in ("utf-8", "utf-8-sig", None):
        try:
            df = pd.read_csv(PRODUCTIVITY_DATA_CSV, encoding=enc)
            break
        except Exception:
            continue
    else:
        return pd.DataFrame()
    # Колонки: Поле, Регион, Год, Культура, урожайность факт ц/га, урожайность прогноз ц/га, active
    df = df.rename(columns={
        "Поле": "field_name",
        "Год": "year",
        "Культура": "crop_name",
        "урожайность факт ц/га": "yield_fact_centners",
        "урожайность прогноз ц/га": "cropwise_pred_centners",
    })
    name2id = load_field_name_to_id()
    cropname2id = load_crop_name_to_id()

    df["field_name"] = df["field_name"].astype(str).str.strip()
    df["field_id"] = df["field_name"].map(name2id)
    df["crop_id"] = df["crop_name"].astype(str).str.strip().map(cropname2id)

    # Урожайность: в файле уже в т/га (несмотря на название колонки ц/га — значения 20–40 соответствуют т/га)
    df["target_yield_t_ha"] = pd.to_numeric(df["yield_fact_centners"], errors="coerce")
    df["cropwise_pred_t_ha"] = pd.to_numeric(df["cropwise_pred_centners"], errors="coerce")

    # Оставляем только строки с валидным field_id и прогнозом Cropwise
    df = df.dropna(subset=["field_id", "cropwise_pred_t_ha"])
    df["field_id"] = df["field_id"].astype(int)
    df["year"] = df["year"].astype(int)

    # Одна запись на (field_id, year): берём последнюю по индексу (финальный прогноз)
    df = df.drop_duplicates(subset=["field_id", "year"], keep="last")
    return df[["field_id", "year", "crop_id", "target_yield_t_ha", "cropwise_pred_t_ha"]].copy()


def prepare_model_predictions() -> pd.DataFrame:
    """
    Предсказания нашей модели для расчёта метрик по культурам.

    В Iteration 2 приоритетно используем OOF (out-of-fold) предсказания
    от регуляризованного GBDT:
      models/oof_predictions_full_extended_all_no_leak_reg.csv
    Если файла нет, пытаемся взять базовый OOF:
      models/oof_predictions_full_extended_all_no_leak.csv
    В крайнем случае считаем in-sample предсказания (НЕ для статьи).
    """
    # 1) Регуляризованный GBDT (Iteration 2)
    if OOF_PRED_REG_CSV.exists():
        df = pd.read_csv(OOF_PRED_REG_CSV)
        out = df[["field_id", "year", "crop_id", "target_yield_t_ha"]].copy()
        out["model_pred_t_ha"] = df["model_pred_oof_t_ha"]
        out["pred_kind"] = "oof_reg"
        return out

    # 2) Базовый OOF (Iteration 1)
    if OOF_PRED_BASE_CSV.exists():
        df = pd.read_csv(OOF_PRED_BASE_CSV)
        out = df[["field_id", "year", "crop_id", "target_yield_t_ha"]].copy()
        out["model_pred_t_ha"] = df["model_pred_oof_t_ha"]
        out["pred_kind"] = "oof"
        return out

    # Fallback: если OOF файла нет, считаем in-sample (НЕ для статьи)
    if not ML_FULL_EXTENDED_CSV.exists():
        print("[WARN] Нет ml_dataset_full_extended.csv. Пропуск предсказаний.")
        return pd.DataFrame()

    print("[WARN] OOF predictions не найдены. Считаю in-sample предсказания (метрики будут завышены).")
    from train_baseline_model import prepare_features, build_pipeline

    df = pd.read_csv(ML_FULL_EXTENDED_CSV)
    X, y, _, feature_cols = prepare_features(df, model_type="all", exclude_leaky=True)
    pipe = build_pipeline(feature_cols)
    pipe.fit(X, y.to_numpy(dtype=float))
    pred = pipe.predict(X)

    out = df[["field_id", "year", "crop_id", "target_yield_t_ha"]].copy()
    out["model_pred_t_ha"] = pred
    out["pred_kind"] = "train_in_sample"
    return out


def build_comparison_dataset() -> pd.DataFrame:
    """Объединяет предсказания модели и Cropwise по (field_id, year). Таргет — из нашего датасета."""
    model_df = prepare_model_predictions()
    cropwise_df = prepare_cropwise_dataset()

    if model_df.empty:
        return pd.DataFrame()
    if cropwise_df.empty:
        model_df["cropwise_pred_t_ha"] = np.nan
        return model_df

    # Мержим по (field_id, year). Таргет и crop_id берём из нашей выборки (model_df).
    cropwise_flat = cropwise_df[["field_id", "year", "cropwise_pred_t_ha"]].drop_duplicates()
    merged = model_df.merge(cropwise_flat, on=["field_id", "year"], how="left")
    return merged


def metrics_one(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    R², RMSE, MAE, MAPE для одной пары (y_true, y_pred).
    Пропуски отбрасываем. R² нестабилен при очень малом N, поэтому считаем его только при N>=5.
    """
    mask = y_true.notna() & y_pred.notna() & (y_true > 0)
    n = int(mask.sum())
    if n == 0:
        return {"N_used": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

    t, p = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(t, p)))
    mae = float(mean_absolute_error(t, p))
    mape = float((np.abs((t - p) / t).mean() * 100))
    r2 = float(r2_score(t, p)) if n >= 5 else np.nan

    return {"N_used": n, "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def compute_metrics_by_crop(comp: pd.DataFrame, crops_df: pd.DataFrame, group_fn) -> pd.DataFrame:
    """Метрики по каждой культуре (crop_id) и по crop_group."""
    rows = []
    comp = comp.copy()
    comp["crop_group"] = comp["crop_id"].apply(group_fn)

    for crop_id, grp in comp.groupby("crop_id"):
        if grp["target_yield_t_ha"].notna().sum() < 2:
            continue
        m_model = metrics_one(grp["target_yield_t_ha"], grp["model_pred_t_ha"])
        m_cw = metrics_one(grp["target_yield_t_ha"], grp["cropwise_pred_t_ha"])
        crop_name = ""
        if crops_df is not None and not crops_df.empty and "id" in crops_df.columns and "name" in crops_df.columns:
            match = crops_df[np.isclose(crops_df["id"].astype(float), float(crop_id))]
            if not match.empty:
                crop_name = str(match["name"].iloc[0])
        rows.append({
            "crop_id": crop_id,
            "crop_name": crop_name,
            "crop_group": group_fn(crop_id),
            "N_obs": len(grp),
            "N_used_model": m_model["N_used"],
            "R2_model": m_model["R2"],
            "RMSE_model": m_model["RMSE"],
            "MAPE_model": m_model["MAPE"],
            "N_used_cropwise": m_cw["N_used"],
            "R2_cropwise": m_cw["R2"],
            "RMSE_cropwise": m_cw["RMSE"],
            "MAPE_cropwise": m_cw["MAPE"],
        })

    return pd.DataFrame(rows)


def compute_metrics_by_group(comp: pd.DataFrame, group_fn) -> pd.DataFrame:
    """Метрики по укрупнённым группам культур."""
    rows = []
    comp = comp.copy()
    comp["crop_group"] = comp["crop_id"].apply(group_fn)

    for cg, grp in comp.groupby("crop_group"):
        if grp["target_yield_t_ha"].notna().sum() < 2:
            continue
        m_model = metrics_one(grp["target_yield_t_ha"], grp["model_pred_t_ha"])
        m_cw = metrics_one(grp["target_yield_t_ha"], grp["cropwise_pred_t_ha"])
        rows.append({
            "crop_group": cg,
            "N_obs": len(grp),
            "N_used_model": m_model["N_used"],
            "R2_model": m_model["R2"],
            "RMSE_model": m_model["RMSE"],
            "MAPE_model": m_model["MAPE"],
            "N_used_cropwise": m_cw["N_used"],
            "R2_cropwise": m_cw["R2"],
            "RMSE_cropwise": m_cw["RMSE"],
            "MAPE_cropwise": m_cw["MAPE"],
        })

    return pd.DataFrame(rows)


def _cropwise_is_same_as_target(by_group: pd.DataFrame) -> bool:
    """Проверка: прогноз Cropwise совпадает с таргетом (R2≈1, RMSE≈0) — не независимый прогноз."""
    if by_group.empty or by_group["R2_cropwise"].isna().all():
        return False
    return (by_group["R2_cropwise"].dropna() >= 0.999).all()


def draft_text_for_article(by_crop: pd.DataFrame, by_group: pd.DataFrame, pred_kind: str) -> str:
    """Черновой текст подпункта 5.X для статьи."""
    cropwise_fake = _cropwise_is_same_as_target(by_group)

    if pred_kind in ("oof", "oof_reg"):
        pred_note = "out-of-fold (OOF) предсказаниях при GroupKFold по годам"
    else:
        pred_note = "обучающей выборке (in-sample)"

    lines = [
        "## 5.X. Качество модели по культурам" + (" и сравнение с Cropwise" if not cropwise_fake else ""),
        "",
        "Средние метрики по всему датасету маскируют существенные различия между культурами. "
        "В таблице X представлены R² и MAPE для предлагаемой модели (full_extended + all_no_leak) в разрезе по культурам и укрупнённым группам (зерновые, масличные, картофель, сахарная свёкла, прочие)."
        "",
        f"Метрики по культурам посчитаны на {pred_note}.",
        "",
    ]

    if cropwise_fake:
        lines.append(
            "В имеющемся экспорте productivity_data значения «прогноз Cropwise» для пересекающихся полей и лет совпадают с фактической урожайностью (таргетом), поэтому честное сравнение с независимым прогнозом Cropwise по этим данным невозможно; в таблице приведены только метрики нашей модели. "
            "Для корректного сравнения с Cropwise необходим экспорт именно предуборочного прогноза."
        )
        lines.append("")

    if not by_group.empty:
        # Для выбора "лучшей" группы игнорируем слишком маленькие группы, где R² неустойчив.
        stable = by_group[(by_group["N_used_model"] >= 5) & by_group["R2_model"].notna()].copy()
        if stable.empty:
            stable = by_group.copy()

        best_r2_model = stable.loc[stable["R2_model"].idxmax()] if stable["R2_model"].notna().any() else stable.iloc[0]
        best_mape_model = stable.loc[stable["MAPE_model"].idxmin()] if stable["MAPE_model"].notna().any() else stable.iloc[0]
        lines.append(
            f"По группам культур наилучший R² у нашей модели достигается для группы {best_r2_model['crop_group']} "
            f"(R² = {best_r2_model['R2_model']:.3f}, N = {int(best_r2_model['N_used_model'])}), "
            f"наименьший MAPE — для группы {best_mape_model['crop_group']} "
            f"(MAPE = {best_mape_model['MAPE_model']:.1f}%, N = {int(best_mape_model['N_used_model'])}). "
        )
        if not cropwise_fake:
            cw_better = by_group[by_group["R2_cropwise"] > by_group["R2_model"]]
            if not cw_better.empty:
                lines.append(
                    f"Прогнозы Cropwise дают более высокий R², чем наша модель, в группах: "
                    + ", ".join(cw_better["crop_group"].astype(str)) + ". "
                )
            else:
                lines.append("По всем группам культур наша модель имеет не меньший R², чем прогнозы Cropwise. ")
        lines.append("")

    lines.append(
        "По отдельным культурам с малым числом наблюдений метрики (особенно R²) могут быть неустойчивыми, поэтому их следует интерпретировать осторожно."
    )
    lines.append("")
    lines.append(
        "Таким образом, разрез по культурам необходим для интерпретации качества модели; усреднение «по больнице» маскирует различия между культурами."
    )
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Подготовка датасета сравнения ===")
    comp = build_comparison_dataset()
    if comp.empty:
        print("Нет данных для сравнения. Завершение.")
        return

    # Пояснение по типу предсказаний (OOF vs in-sample)
    if "pred_kind" in comp.columns:
        print("Тип предсказаний нашей модели:", comp["pred_kind"].dropna().unique().tolist())

    # Таргет для сравнения — из нашего датасета (target_yield_t_ha уже из model_df)
    comp = comp[comp["target_yield_t_ha"].notna() & (comp["target_yield_t_ha"] > 0)]
    comp.to_csv(OUT_COMPARISON_CSV, index=False)
    print(f"Сохранено: {OUT_COMPARISON_CSV}, строк: {len(comp)}")

    # Cropwise есть только где есть пересечение с productivity_data
    has_cropwise = comp["cropwise_pred_t_ha"].notna().sum()
    print(f"Строк с прогнозом Cropwise: {has_cropwise}")

    print("\n=== 2. Группы культур и метрики по культурам/группам ===")
    crops_df = pd.read_csv(CROPS_CSV, encoding="utf-8") if CROPS_CSV.exists() else None
    group_fn = build_crop_group_mapping(crops_df if crops_df is not None else pd.DataFrame())

    by_crop = compute_metrics_by_crop(comp, crops_df, group_fn)
    by_group = compute_metrics_by_group(comp, group_fn)

    by_crop.to_csv(OUT_METRICS_BY_CROP_CSV, index=False)
    by_group.to_csv(OUT_METRICS_BY_GROUP_CSV, index=False)
    print(f"Сохранено: {OUT_METRICS_BY_CROP_CSV}, {OUT_METRICS_BY_GROUP_CSV}")

    if _cropwise_is_same_as_target(by_group):
        print("\n[!] Внимание: «прогноз Cropwise» в данных совпадает с таргетом (R2=1, RMSE=0). "
              "Это не независимый прогноз — сравнение с Cropwise в статье не интерпретировать как сравнение двух моделей.")

    print("\n--- Метрики по группам культур ---")
    print(by_group.to_string(index=False))

    print("\n--- Метрики по культурам (топ по N_obs) ---")
    print(by_crop.nlargest(15, "N_obs").to_string(index=False))

    print("\n=== 3. Черновой текст для раздела 5 ===")
    pred_kind = "unknown"
    if "pred_kind" in comp.columns:
        kinds = comp["pred_kind"].dropna().unique().tolist()
        pred_kind = kinds[0] if kinds else "unknown"
    draft = draft_text_for_article(by_crop, by_group, pred_kind=str(pred_kind))
    with open(OUT_DRAFT_TEXT_MD, "w", encoding="utf-8") as f:
        f.write(draft)
    print(f"Сохранено: {OUT_DRAFT_TEXT_MD}")
    print(draft)


if __name__ == "__main__":
    main()
