"""
Скрипт для сборки финального датасета для ML-модели.

Объединяет:
- Базовые фичи из полей и операций
- Расширенные NDVI фичи (агрегации по сезону, фазам, тренды, зеленый период)
- Расширенные фичи из операций (NPK из application_mix_items, временные интервалы)
- Таргет (урожайность из productivity_estimates)
"""

import os
import json
import ast
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Always run relative to this file (fixes Windows cwd/path issues)
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data_raw"
OUTPUT_DIR = "data_processed"

# Input files
FIELDS_CSV = os.path.join(DATA_DIR, "fields.csv")
CROPS_CSV = os.path.join(DATA_DIR, "crops.csv")
OPERATIONS_CSV = os.path.join(DATA_DIR, "operations.csv")
PRODUCTIVITY_CSV = os.path.join(DATA_DIR, "productivity_estimates.csv")
NDVI_CSV = os.path.join(DATA_DIR, "ndvi_timeseries.csv")
ADDITIONAL_YIELDS_CSV = os.path.join(DATA_DIR, "additional_yields_from_history.csv")
FERTILIZERS_NPK_CSV = os.path.join(DATA_DIR, "fertilizers_npk.csv")
HISTORY_ITEMS_CSV = os.path.join(DATA_DIR, "history_items_full.csv")
YIELD_MAPS_CSV = os.path.join(DATA_DIR, "yield_maps.csv")
SOIL_TESTS_CSV = os.path.join(DATA_DIR, "soil_tests.csv")
WEATHER_SEASON_AGG_CSV = os.path.join(DATA_DIR, "weather_season_aggregates.csv")  # legacy aggregates
WEATHER_HISTORY_ITEMS_CSV = os.path.join(DATA_DIR, "weather_history_items.csv")  # daily weather by field_group_id

# Output files
ML_DATASET_CSV = os.path.join(OUTPUT_DIR, "ml_dataset_with_ndvi.csv")
TARGETS_EXTENDED_CSV = os.path.join(OUTPUT_DIR, "targets_extended.csv")
ML_DATASET_NDVI_ONLY_EXT_CSV = os.path.join(OUTPUT_DIR, "ml_dataset_ndvi_only_extended.csv")
ML_DATASET_OPS_NDVI_EXT_CSV = os.path.join(OUTPUT_DIR, "ml_dataset_ops_ndvi_2021_2025.csv")
ML_DATASET_FULL_EXT_CSV = os.path.join(OUTPUT_DIR, "ml_dataset_full_extended.csv")

# Versioned outputs for crop-window NDVI/weather (do NOT overwrite legacy datasets)
CROP_WINDOW_TAG = "cropwindow_v1_may_sep_oct"
ML_DATASET_NDVI_ONLY_EXT_CROP_WINDOW_CSV = os.path.join(
    OUTPUT_DIR, f"ml_dataset_ndvi_only_extended_{CROP_WINDOW_TAG}.csv"
)
ML_DATASET_OPS_NDVI_EXT_CROP_WINDOW_CSV = os.path.join(
    OUTPUT_DIR, f"ml_dataset_ops_ndvi_2021_2025_{CROP_WINDOW_TAG}.csv"
)
ML_DATASET_FULL_EXT_CROP_WINDOW_CSV = os.path.join(
    OUTPUT_DIR, f"ml_dataset_full_extended_{CROP_WINDOW_TAG}.csv"
)

# If True, script will also rebuild legacy CSVs (overwriting old files). Keep False by default.
REBUILD_LEGACY_OUTPUTS = False

# ============================================================================
# CROP-SEASON WINDOWS (crop-dependent)
# ============================================================================
#
# These features are computed inside crop-specific seasonal windows.
# If crop is unknown, we use DEFAULT_* values.
#
DEFAULT_SEASON_START_MONTH = 5  # May
DEFAULT_SEASON_END_MONTH = 10   # October

# Mapping from crop "kind" -> season window (months)
CROP_SEASON_START = {
    "spring_wheat": 5,  # May
    "sunflower": 5,
}

CROP_SEASON_END = {
    "spring_wheat": 9,  # September
    "sunflower": 10,
}

# Base temperature for GDD by crop kind
CROP_GDD_BASE_TEMP = {
    "spring_wheat": 5.0,
    "sunflower": 10.0,
}

# Map Cropwise crops.standard_name -> our crop kind keys above
# (crops.csv already contains standard_name like wheat_spring, sunflower, etc.)
STANDARD_NAME_TO_CROP_KIND = {
    "wheat_spring": "spring_wheat",
    "sunflower": "sunflower",
}

# Если вдруг в application_mix_items появится applicable_name — используем маппинг (иначе fallback на грубую оценку)
FERTILIZER_NPK_MAP: Dict[str, Dict[str, float]] = {
    # Примеры (заполни/расширь по своим данным, если applicable_name станет доступен)
    "Аммиачная селитра": {"N": 0.34, "P": 0.0, "K": 0.0},
    "Мочевина": {"N": 0.46, "P": 0.0, "K": 0.0},
    "Диаммофоска": {"N": 0.10, "P": 0.26, "K": 0.26},
}

# Если есть data_raw/fertilizers_npk.csv — будем использовать точный состав NPK по id
FERTILIZERS_NPK_BY_ID: Dict[int, Dict[str, float]] = {}


def load_fertilizers_npk() -> None:
    """
    Заполняет глобальный словарь FERTILIZERS_NPK_BY_ID из data_raw/fertilizers_npk.csv (если он есть).
    Ожидает колонки: id, N, P2O5, K2O (в процентах по массе).
    """
    global FERTILIZERS_NPK_BY_ID
    if not os.path.exists(FERTILIZERS_NPK_CSV):
        return
    try:
        fert_df = pd.read_csv(FERTILIZERS_NPK_CSV)
    except Exception:
        return
    required = {"id", "N", "P2O5", "K2O"}
    if not required.issubset(fert_df.columns):
        return
    mapping: Dict[int, Dict[str, float]] = {}
    for _, r in fert_df.dropna(subset=["id"]).iterrows():
        try:
            fid = int(r["id"])
        except Exception:
            continue
        mapping[fid] = {
            "N": float(r.get("N", 0.0)),
            "P2O5": float(r.get("P2O5", 0.0)),
            "K2O": float(r.get("K2O", 0.0)),
        }
    FERTILIZERS_NPK_BY_ID = mapping


# ============================================================================
# SOIL FEATURES (SoilTest)
# ============================================================================

def build_soil_features() -> pd.DataFrame:
    """
    Строит soil-фичи уровня поля из data_raw/soil_tests.csv.

    Для каждого field_id берём последний по made_at SoilTest и оставляем
    только нужные показатели, переименованные с префиксом field_soil_*.
    """
    if not os.path.exists(SOIL_TESTS_CSV):
        print("ℹ️ soil_tests.csv не найден — field_soil_* фичи недоступны")
        return pd.DataFrame(columns=["field_id"])

    try:
        soil_df = pd.read_csv(SOIL_TESTS_CSV)
    except Exception as e:
        print(f"⚠️ Не удалось прочитать soil_tests.csv: {e}")
        return pd.DataFrame(columns=["field_id"])

    if "field_id" not in soil_df.columns:
        print("⚠️ В soil_tests.csv нет колонки field_id — пропускаю soil-фичи")
        return pd.DataFrame(columns=["field_id"])

    # Приводим made_at к дате и выбираем последний тест по полю
    if "made_at" in soil_df.columns:
        soil_df["made_at_parsed"] = pd.to_datetime(
            soil_df["made_at"], errors="coerce", utc=True
        ).dt.tz_convert(None)
    else:
        soil_df["made_at_parsed"] = pd.NaT

    soil_df["field_id"] = pd.to_numeric(soil_df["field_id"], errors="coerce").astype("Int64")
    soil_df = soil_df.dropna(subset=["field_id"]).copy()

    if len(soil_df) == 0:
        return pd.DataFrame(columns=["field_id"])

    soil_df_sorted = soil_df.sort_values(["field_id", "made_at_parsed"])
    latest_soil = soil_df_sorted.groupby("field_id").tail(1).copy()

    col_map = {
        "soil_pH.value": "field_soil_pH",
        "soil_organic_matter.value": "field_soil_OM",
        "soil_P.value": "field_soil_P",
        "soil_K.value": "field_soil_K",
        "soil_N.value": "field_soil_N",
        "soil_N_NO3.value": "field_soil_N_NO3",
        "soil_Mg.value": "field_soil_Mg",
        "soil_cation_exchange_capacity.value": "field_soil_CEC",
        "soil_Ca_saturation.value": "field_soil_Ca_saturation",
    }

    cols_present = ["field_id"]
    for src, dst in col_map.items():
        if src in latest_soil.columns:
            cols_present.append(src)

    soil_features = latest_soil[cols_present].copy()
    rename_dict = {src: dst for src, dst in col_map.items() if src in soil_features.columns}
    soil_features = soil_features.rename(columns=rename_dict)

    # Если есть и N, и N_NO3, то используем N как основной, а N_NO3 оставляем отдельной фичей
    if "field_soil_N" not in soil_features.columns and "field_soil_N_NO3" in soil_features.columns:
        soil_features = soil_features.rename(columns={"field_soil_N_NO3": "field_soil_N"})

    print(f"  ✅ Soil-фичи: {len(soil_features)} полей, колонок: {len(soil_features.columns) - 1}")
    return soil_features


# ============================================================================
# WEATHER FEATURES (weather_season_aggregates)
# ============================================================================

def _build_crop_id_to_standard_name(crops_df: pd.DataFrame) -> dict[int, str]:
    """Build crop_id -> standard_name mapping from crops.csv."""
    if crops_df is None or crops_df.empty:
        return {}
    if "id" not in crops_df.columns or "standard_name" not in crops_df.columns:
        return {}
    tmp = crops_df[["id", "standard_name"]].dropna(subset=["id"]).copy()
    out: dict[int, str] = {}
    for _, r in tmp.iterrows():
        try:
            cid = int(float(r["id"]))
        except Exception:
            continue
        std = str(r.get("standard_name") or "").strip()
        if std:
            out[cid] = std
    return out


def _season_params_for_crop_id(crop_id: object, crop_id_to_std: dict[int, str]) -> tuple[int, int, float]:
    """
    Resolve (season_start_month, season_end_month, gdd_base_temp) for given crop_id.
    Unknown crop -> defaults.
    """
    start = DEFAULT_SEASON_START_MONTH
    end = DEFAULT_SEASON_END_MONTH
    base_temp = float(CROP_GDD_BASE_TEMP.get("sunflower", 10.0))

    if crop_id is None or (isinstance(crop_id, float) and np.isnan(crop_id)):
        return start, end, base_temp

    try:
        cid = int(float(crop_id))
    except Exception:
        return start, end, base_temp

    std = crop_id_to_std.get(cid, "")
    kind = STANDARD_NAME_TO_CROP_KIND.get(std, "")
    if kind:
        start = int(CROP_SEASON_START.get(kind, start))
        end = int(CROP_SEASON_END.get(kind, end))
        base_temp = float(CROP_GDD_BASE_TEMP.get(kind, base_temp))
    return start, end, base_temp


def build_weather_features_daily(
    fields_df: pd.DataFrame,
    keys_df: pd.DataFrame,
    crop_id_to_std: dict[int, str],
) -> pd.DataFrame:
    """
    Build compact crop-window weather features (field_id, year) from daily weather_history_items.csv.

    Features (computed inside crop-specific season window months):
      - weather_temp_avg_season: mean temp_avg in season
      - weather_gdd_season: sum(max(temp_avg - base_temp, 0))
      - weather_precip_sum_season: sum precipitation in season
      - weather_precip_sum_early: precipitation in first half of season (by months)
      - weather_hot_days: count of days with temp_max > 30C

    NOTE: weather_history_items.csv is keyed by field_group_id (not field_id),
    so we map field_id -> field_group_id via fields.csv.
    """
    required_keys = {"field_id", "year"}
    if not required_keys.issubset(keys_df.columns):
        return pd.DataFrame(columns=["field_id", "year"])
    if "id" not in fields_df.columns or "field_group_id" not in fields_df.columns:
        print("⚠️ В fields.csv нет id/field_group_id — не могу маппить погоду на поля")
        return pd.DataFrame(columns=["field_id", "year"])

    if not os.path.exists(WEATHER_HISTORY_ITEMS_CSV):
        print("ℹ️ weather_history_items.csv не найден — crop-window weather_* фичи недоступны")
        return pd.DataFrame(columns=["field_id", "year"])

    try:
        w = pd.read_csv(WEATHER_HISTORY_ITEMS_CSV)
    except Exception as e:
        print(f"⚠️ Не удалось прочитать weather_history_items.csv: {e}")
        return pd.DataFrame(columns=["field_id", "year"])

    needed_cols = {"date", "temperature_avg", "temperature_max", "precipitation", "field_group_id", "year"}
    if not needed_cols.issubset(set(w.columns)):
        print(f"⚠️ weather_history_items.csv без нужных колонок {sorted(list(needed_cols))} — пропускаю weather-фичи")
        return pd.DataFrame(columns=["field_id", "year"])

    # field_id -> field_group_id
    fmap = fields_df[["id", "field_group_id"]].copy()
    fmap["id"] = pd.to_numeric(fmap["id"], errors="coerce").astype("Int64")
    fmap["field_group_id"] = pd.to_numeric(fmap["field_group_id"], errors="coerce").astype("Int64")
    fmap = fmap.dropna(subset=["id", "field_group_id"]).copy()
    field_to_group = fmap.set_index("id")["field_group_id"].to_dict()

    # parse weather
    w = w.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w["year"] = pd.to_numeric(w["year"], errors="coerce").astype("Int64")
    w["field_group_id"] = pd.to_numeric(w["field_group_id"], errors="coerce").astype("Int64")
    w["temperature_avg"] = pd.to_numeric(w["temperature_avg"], errors="coerce")
    w["temperature_max"] = pd.to_numeric(w["temperature_max"], errors="coerce")
    w["precipitation"] = pd.to_numeric(w["precipitation"], errors="coerce")

    w = w.dropna(subset=["date", "year", "field_group_id"]).copy()
    w["month"] = w["date"].dt.month.astype(int)

    # keys_df: enforce types
    keys = keys_df[["field_id", "year", "crop_id"]].copy() if "crop_id" in keys_df.columns else keys_df[["field_id", "year"]].assign(crop_id=np.nan)
    keys["field_id"] = pd.to_numeric(keys["field_id"], errors="coerce").astype("Int64")
    keys["year"] = pd.to_numeric(keys["year"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["field_id", "year"]).copy()
    keys["field_id"] = keys["field_id"].astype(int)
    keys["year"] = keys["year"].astype(int)

    out_rows: list[dict] = []

    # Pre-index weather by (field_group_id, year) for speed
    w_groups = w.groupby(["field_group_id", "year"], sort=False)

    for _, r in keys.iterrows():
        fid = int(r["field_id"])
        yr = int(r["year"])
        crop_id = r.get("crop_id", np.nan)

        fg = field_to_group.get(fid)
        if fg is None:
            out_rows.append({
                "field_id": fid,
                "year": yr,
                "weather_temp_avg_season": np.nan,
                "weather_gdd_season": np.nan,
                "weather_precip_sum_season": np.nan,
                "weather_precip_sum_early": np.nan,
                "weather_hot_days": np.nan,
            })
            continue

        try:
            w_one = w_groups.get_group((int(fg), yr)).copy()
        except KeyError:
            w_one = pd.DataFrame()

        start_m, end_m, base_temp = _season_params_for_crop_id(crop_id, crop_id_to_std)
        # early half by months (inclusive)
        mid_m = int((start_m + end_m) / 2)

        if w_one.empty:
            out_rows.append({
                "field_id": fid,
                "year": yr,
                "weather_temp_avg_season": np.nan,
                "weather_gdd_season": np.nan,
                "weather_precip_sum_season": np.nan,
                "weather_precip_sum_early": np.nan,
                "weather_hot_days": np.nan,
            })
            continue

        # filter to crop season window months
        w_season = w_one[(w_one["month"] >= start_m) & (w_one["month"] <= end_m)].copy()
        if w_season.empty:
            out_rows.append({
                "field_id": fid,
                "year": yr,
                "weather_temp_avg_season": np.nan,
                "weather_gdd_season": np.nan,
                "weather_precip_sum_season": np.nan,
                "weather_precip_sum_early": np.nan,
                "weather_hot_days": np.nan,
            })
            continue

        tavg = w_season["temperature_avg"]
        tmax = w_season["temperature_max"]
        pr = w_season["precipitation"].fillna(0.0)

        weather_temp_avg_season = float(tavg.mean()) if tavg.notna().any() else np.nan
        gdd = (tavg - base_temp).clip(lower=0.0)
        weather_gdd_season = float(gdd.sum()) if gdd.notna().any() else np.nan
        weather_precip_sum_season = float(pr.sum())

        w_early = w_season[(w_season["month"] >= start_m) & (w_season["month"] <= mid_m)].copy()
        weather_precip_sum_early = float(w_early["precipitation"].fillna(0.0).sum()) if not w_early.empty else 0.0

        hot_days = (tmax > 30.0).sum() if tmax.notna().any() else np.nan

        out_rows.append({
            "field_id": fid,
            "year": yr,
            "weather_temp_avg_season": weather_temp_avg_season,
            "weather_gdd_season": weather_gdd_season,
            "weather_precip_sum_season": weather_precip_sum_season,
            "weather_precip_sum_early": weather_precip_sum_early,
            "weather_hot_days": float(hot_days) if not pd.isna(hot_days) else np.nan,
        })

    wfeat = pd.DataFrame(out_rows)
    wfeat = wfeat.drop_duplicates(subset=["field_id", "year"], keep="last")
    print(f"  ✅ Weather (daily, crop-window): {len(wfeat)} field×year, колонок: {len(wfeat.columns) - 2}")
    return wfeat


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_application_mix_items(mix_str: str) -> Dict[str, float]:
    """
    Парсит JSON строку application_mix_items и извлекает NPK.
    
    Returns:
        dict с ключами 'npk_n', 'npk_p', 'npk_k' и суммарными нормами
    """
    if pd.isna(mix_str) or mix_str == '' or mix_str == '[]':
        return {'npk_n': 0.0, 'npk_p': 0.0, 'npk_k': 0.0}
    
    try:
        # application_mix_items в CSV часто записан как python-literal со строками в single quotes
        # Сначала пробуем json, потом ast.literal_eval
        if isinstance(mix_str, str):
            try:
                items = json.loads(mix_str)
            except json.JSONDecodeError:
                items = ast.literal_eval(mix_str)
        else:
            items = mix_str
        if not isinstance(items, list):
            return {'npk_n': 0.0, 'npk_p': 0.0, 'npk_k': 0.0}
        
        npk_n, npk_p, npk_k = 0.0, 0.0, 0.0
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            # Ищем NPK в разных полях
            applicable_type = item.get('applicable_type', '')
            fact_rate = item.get('fact_rate', 0) or item.get('planned_rate', 0) or item.get('value', 0)
            applicable_name = item.get('applicable_name')  # чаще всего отсутствует в твоих данных
            applicable_id = item.get('applicable_id')
            
            # Упрощенная логика: если это Fertilizer, считаем что может содержать NPK
            # В идеале маппим applicable_id -> fertilizers.id из fertilizers_npk.csv
            if applicable_type == 'Fertilizer' and fact_rate > 0:
                used_precise = False
                # 1) Пытаемся использовать точный состав из FERTILIZERS_NPK_BY_ID (если загружен)
                try:
                    if applicable_id is not None:
                        fid = int(applicable_id)
                        comp_id = FERTILIZERS_NPK_BY_ID.get(fid)
                        if comp_id:
                            # N, P2O5, K2O — проценты, переводим в "доля" (0–1)
                            n_frac = float(comp_id.get("N", 0.0)) / 100.0
                            p_frac = float(comp_id.get("P2O5", 0.0)) / 100.0
                            k_frac = float(comp_id.get("K2O", 0.0)) / 100.0
                            rate = float(fact_rate)
                            npk_n += rate * n_frac
                            npk_p += rate * p_frac
                            npk_k += rate * k_frac
                            used_precise = True
                except Exception:
                    used_precise = False

                # 2) Фоллбек по имени (старый FERTILIZER_NPK_MAP)
                if not used_precise:
                    name = str(applicable_name).strip() if applicable_name else ""
                    if name and name in FERTILIZER_NPK_MAP:
                        comp = FERTILIZER_NPK_MAP[name]
                        rate = float(fact_rate)
                        npk_n += rate * float(comp.get("N", 0.0))
                        npk_p += rate * float(comp.get("P", 0.0))
                        npk_k += rate * float(comp.get("K", 0.0))
                    else:
                        # 3) Фоллбек: грубая оценка 30/10/10, если ни id, ни имя не помогли
                        rate = float(fact_rate)
                        npk_n += rate * 0.3  # Примерная доля N
                        npk_p += rate * 0.1  # Примерная доля P
                        npk_k += rate * 0.1  # Примерная доля K
        
        return {'npk_n': npk_n, 'npk_p': npk_p, 'npk_k': npk_k}
    except (json.JSONDecodeError, ValueError, SyntaxError, TypeError, AttributeError):
        return {'npk_n': 0.0, 'npk_p': 0.0, 'npk_k': 0.0}


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_baseline_features(
    fields_df: pd.DataFrame,
    crops_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    productivity_df: pd.DataFrame,
    seeds_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Создает базовые фичи из полей, культур и операций для каждого поле×год.
    
    Включает:
    - field_*: геометрия и координаты полей
    - crop_*: информация о культуре
    - ops_*: агрегированные фичи из операций
    """
    features = []

    # Нормализуем season -> int (в CSV может быть и int, и str)
    ops = operations_df.copy()
    ops['season_year'] = pd.to_numeric(ops.get('season'), errors='coerce')

    def _parse_dt_series(s: pd.Series) -> pd.Series:
        """
        Приводит даты к единому виду: tz-naive Timestamp в UTC.
        Это нужно, чтобы можно было безопасно сравнивать planned_start_date (naive)
        и actual_start_datetime (tz-aware).
        """
        dt = pd.to_datetime(s, errors='coerce', utc=True)
        # делаем tz-naive (в UTC)
        return dt.dt.tz_convert(None)

    def _is_seeding_row(df: pd.DataFrame) -> pd.Series:
        """
        В operations.csv посев часто лежит в operation_type=application (в application_mix_items есть Seed),
        а не в отдельном operation_type=seeding.
        """
        if 'operation_type' in df.columns and (df['operation_type'] == 'seeding').any():
            return df['operation_type'] == 'seeding'
        if 'application_mix_items' in df.columns:
            s = df['application_mix_items'].astype(str)
            return s.str.contains("applicable_type': 'Seed'", na=False) | s.str.contains('\"applicable_type\": \"Seed\"', na=False)
        return pd.Series([False] * len(df), index=df.index)
    
    def _extract_seed_ids_from_mix(mix_raw: object) -> list[int]:
        """
        Возвращает список seed_id (applicable_id) из application_mix_items для applicable_type == 'Seed'.
        application_mix_items у тебя — Python-literal string, парсим через ast.literal_eval.
        """
        if pd.isna(mix_raw) or str(mix_raw).strip() in ("", "[]", "nan", "None"):
            return []
        try:
            try:
                items = json.loads(mix_raw)
            except Exception:
                items = ast.literal_eval(mix_raw)
        except Exception:
            return []
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return []
        seed_ids: list[int] = []
        for it in items:
            if isinstance(it, dict) and it.get("applicable_type") == "Seed" and it.get("applicable_id") is not None:
                try:
                    seed_ids.append(int(it["applicable_id"]))
                except Exception:
                    continue
        return seed_ids

    # seed_id -> crop_id справочник (если seeds_df доступен)
    seed_to_crop: dict[int, int] = {}
    if seeds_df is not None and len(seeds_df) > 0 and "id" in seeds_df.columns and "crop_id" in seeds_df.columns:
        tmp = seeds_df[["id", "crop_id"]].dropna()
        try:
            seed_to_crop = {int(r["id"]): int(r["crop_id"]) for _, r in tmp.iterrows()}
        except Exception:
            seed_to_crop = {}

    for _, row in productivity_df.iterrows():
        field_id = int(row['field_id'])
        year = int(row['year'])
        
        # Информация о поле
        field_info = fields_df[fields_df['id'] == field_id]
        if len(field_info) == 0:
            continue
        
        field = field_info.iloc[0]
        
        # Операции для этого поля и года
        field_ops = ops[
            (ops['field_id'] == field_id) &
            (ops['season_year'] == year)
        ].copy()
        
        # ====================================================================
        # FIELD FEATURES (field_*)
        # ====================================================================
        feature_row = {
            'field_id': field_id,
            'year': year,
            'field_tillable_area': field.get('tillable_area', np.nan),
            'field_calculated_area': field.get('calculated_area', np.nan),
            'field_lat': field.get('lat', np.nan),
            'field_long': field.get('long', np.nan),
        }
        
        # ====================================================================
        # OPERATIONS FEATURES (ops_*)
        # ====================================================================
        if len(field_ops) > 0:
            # Общее количество операций
            feature_row['ops_count_total'] = len(field_ops)
            
            # Количество операций по типам
            op_type_counts = field_ops['operation_type'].value_counts()
            feature_row['ops_count_soil'] = op_type_counts.get('soil', 0)
            feature_row['ops_count_application'] = op_type_counts.get('application', 0)
            feature_row['ops_count_harvesting'] = op_type_counts.get('harvesting', 0)
            feature_row['ops_count_seeding'] = op_type_counts.get('seeding', 0)
            feature_row['ops_count_tillage'] = op_type_counts.get('tillage', 0)
            feature_row['ops_count_spraying'] = op_type_counts.get('spraying', 0)
            
            # NPK из application_mix_items
            npk_n_total, npk_p_total, npk_k_total = 0.0, 0.0, 0.0
            if 'application_mix_items' in field_ops.columns:
                for mix_str in field_ops['application_mix_items'].dropna():
                    npk = parse_application_mix_items(mix_str)
                    npk_n_total += npk['npk_n']
                    npk_p_total += npk['npk_p']
                    npk_k_total += npk['npk_k']
            
            feature_row['ops_npk_n_total'] = npk_n_total
            feature_row['ops_npk_p_total'] = npk_p_total
            feature_row['ops_npk_k_total'] = npk_k_total
            feature_row['ops_npk_total'] = npk_n_total + npk_p_total + npk_k_total
            
            # Даты операций
            date_cols = ['planned_start_date', 'actual_start_datetime', 'completed_date']
            all_dates = []
            seeding_dates = []
            harvesting_dates = []
            
            for col in date_cols:
                if col in field_ops.columns:
                    dates = _parse_dt_series(field_ops[col]).dropna()
                    all_dates.extend(dates.tolist())
            
            # Даты посева и уборки
            seeding_ops = field_ops[_is_seeding_row(field_ops)]
            if len(seeding_ops) > 0:
                for col in date_cols:
                    if col in seeding_ops.columns:
                        dates = _parse_dt_series(seeding_ops[col]).dropna()
                        seeding_dates.extend(dates.tolist())
            
            harvesting_ops = field_ops[field_ops['operation_type'] == 'harvesting'] if 'operation_type' in field_ops.columns else field_ops.iloc[0:0]
            if len(harvesting_ops) > 0:
                for col in date_cols:
                    if col in harvesting_ops.columns:
                        dates = _parse_dt_series(harvesting_ops[col]).dropna()
                        harvesting_dates.extend(dates.tolist())
            
            if len(all_dates) > 0:
                feature_row['ops_season_start'] = min(all_dates).strftime('%Y-%m-%d')
                feature_row['ops_season_end'] = max(all_dates).strftime('%Y-%m-%d')
                feature_row['ops_season_duration_days'] = (max(all_dates) - min(all_dates)).days
            else:
                feature_row['ops_season_start'] = None
                feature_row['ops_season_end'] = None
                feature_row['ops_season_duration_days'] = np.nan
            
            # Дни от посева до уборки
            if len(seeding_dates) > 0 and len(harvesting_dates) > 0:
                seeding_date = min(seeding_dates)
                harvesting_date = max(harvesting_dates)
                feature_row['ops_days_seeding_to_harvest'] = (harvesting_date - seeding_date).days
            else:
                feature_row['ops_days_seeding_to_harvest'] = np.nan
            
            # Урожайность из операций уборки
            if len(harvesting_ops) > 0 and 'harvested_weight' in harvesting_ops.columns:
                total_weight = harvesting_ops['harvested_weight'].fillna(0).sum()
                area = feature_row.get('field_tillable_area', 1)
                if area > 0:
                    feature_row['ops_yield_t_ha'] = total_weight / area
                else:
                    feature_row['ops_yield_t_ha'] = np.nan
            else:
                feature_row['ops_yield_t_ha'] = np.nan
            
            # crop_id в твоем operations.csv отсутствует → пытаемся восстановить через /seeds/{id}.crop_id
            feature_row['crop_id'] = np.nan

            if seed_to_crop and "application_mix_items" in field_ops.columns:
                seed_ids_all: list[int] = []
                for mix_raw in field_ops["application_mix_items"].dropna().tolist():
                    seed_ids_all.extend(_extract_seed_ids_from_mix(mix_raw))
                if seed_ids_all:
                    crop_ids = [seed_to_crop.get(sid) for sid in seed_ids_all if seed_to_crop.get(sid) is not None]
                    if crop_ids:
                        feature_row["crop_id"] = int(Counter(crop_ids).most_common(1)[0][0])
        else:
            # Нет операций
            feature_row['ops_count_total'] = 0
            feature_row['ops_count_soil'] = 0
            feature_row['ops_count_application'] = 0
            feature_row['ops_count_harvesting'] = 0
            feature_row['ops_count_seeding'] = 0
            feature_row['ops_count_tillage'] = 0
            feature_row['ops_count_spraying'] = 0
            feature_row['ops_npk_n_total'] = 0
            feature_row['ops_npk_p_total'] = 0
            feature_row['ops_npk_k_total'] = 0
            feature_row['ops_npk_total'] = 0
            feature_row['ops_season_start'] = None
            feature_row['ops_season_end'] = None
            feature_row['ops_season_duration_days'] = np.nan
            feature_row['ops_days_seeding_to_harvest'] = np.nan
            feature_row['ops_yield_t_ha'] = np.nan
            feature_row['crop_id'] = np.nan
        
        # ====================================================================
        # CROP FEATURES (crop_*)
        # ====================================================================
        if not pd.isna(feature_row.get('crop_id')):
            crop_info = crops_df[crops_df['id'] == int(feature_row['crop_id'])]
            if len(crop_info) > 0:
                crop = crop_info.iloc[0]
                feature_row['crop_name'] = crop.get('name', '')
                feature_row['crop_season_type'] = crop.get('season_type', '')
            else:
                feature_row['crop_name'] = ''
                feature_row['crop_season_type'] = ''
        else:
            feature_row['crop_name'] = ''
            feature_row['crop_season_type'] = ''
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


def build_ndvi_features(
    ndvi_df: pd.DataFrame,
    productivity_df: pd.DataFrame,
    operations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates crop-window NDVI features for each field×year.
    
    Фичи:
    - ndvi_max_season: max NDVI within crop season window
    - ndvi_mean_season: mean NDVI within crop season window
    - ndvi_early / ndvi_mid / ndvi_late: mean NDVI by thirds of the season window (by date order)

    Notes:
    - We filter NDVI rows by calendar year == year AND by crop-specific season months.
    - Thirds are defined by splitting observations into 3 near-equal groups (np.array_split).
    """
    features = []

    # Приводим к нужным типам 1 раз (быстрее/чище)
    ndvi_df = ndvi_df.copy()
    # ВАЖНО: фикс tz-aware vs tz-naive — всегда приводим к tz-naive UTC
    ndvi_df['date'] = pd.to_datetime(ndvi_df['date'], errors='coerce', utc=True).dt.tz_convert(None)
    ndvi_df['ndvi_mean'] = pd.to_numeric(ndvi_df['ndvi_mean'], errors='coerce')
    ndvi_df = ndvi_df.dropna(subset=['date', 'ndvi_mean'])

    # ВАЖНО: твой ndvi_timeseries.csv сейчас содержит полный исторический ряд,
    # но разметка year часто не ограничивает даты сезона.
    # Поэтому дополнительно будем фильтровать по calendar year == year.

    # Группировка NDVI по field_id×year (быстрее, чем фильтровать на каждой итерации)
    ndvi_groups = ndvi_df.groupby(['field_id', 'year'], sort=False)

    # Карта даты посева по field×year (для ops_days_seeding_to_ndvi_peak)
    ops = operations_df.copy()
    ops['season_year'] = pd.to_numeric(ops.get('season'), errors='coerce')
    seeding_frames = []
    if 'operation_type' in ops.columns:
        seeding_frames.append(ops[ops['operation_type'] == 'seeding'].copy())
    if 'application_mix_items' in ops.columns:
        seed_mask = ops['application_mix_items'].astype(str).str.contains("applicable_type': 'Seed'", na=False) | \
                    ops['application_mix_items'].astype(str).str.contains('\"applicable_type\": \"Seed\"', na=False)
        seeding_frames.append(ops[seed_mask].copy())

    seeding_ops = pd.concat(seeding_frames, ignore_index=True) if seeding_frames else ops.iloc[0:0].copy()
    if len(seeding_ops) > 0:
        # опорная дата операции: actual_start_datetime -> planned_start_date -> completed_date
        seeding_ops['op_date'] = pd.to_datetime(seeding_ops.get('actual_start_datetime'), errors='coerce', utc=True).dt.tz_convert(None)
        if 'planned_start_date' in seeding_ops.columns:
            seeding_ops['op_date'] = seeding_ops['op_date'].fillna(
                pd.to_datetime(seeding_ops['planned_start_date'], errors='coerce', utc=True).dt.tz_convert(None)
            )
        if 'completed_date' in seeding_ops.columns:
            seeding_ops['op_date'] = seeding_ops['op_date'].fillna(
                pd.to_datetime(seeding_ops['completed_date'], errors='coerce', utc=True).dt.tz_convert(None)
            )
        seeding_ops = seeding_ops.dropna(subset=['op_date', 'season_year'])
        seeding_date_map = (seeding_ops
                            .groupby(['field_id', 'season_year'])['op_date']
                            .min()
                            .to_dict())
    else:
        seeding_date_map = {}

    # Диагностика фаз NDVI: выведем 3-5 примеров реальных field×year
    diag_printed = 0
    diag_max = 5

    # crop mapping for season windows
    crop_id_to_std: dict[int, str] = {}
    if "crops_df" in globals():
        # no-op, kept for mypy
        pass

    # We accept crop_id column in productivity_df (recommended for crop-window features).
    # If it's missing, defaults will be used.
    crop_id_series = productivity_df["crop_id"] if "crop_id" in productivity_df.columns else pd.Series([np.nan] * len(productivity_df))

    for (idx, row), crop_id_val in zip(productivity_df.iterrows(), crop_id_series.tolist()):
        field_id = int(row['field_id'])
        year = int(row['year'])
        
        # Берем группу NDVI
        try:
            field_ndvi = ndvi_groups.get_group((field_id, year)).copy()
        except KeyError:
            field_ndvi = pd.DataFrame()
        
        if len(field_ndvi) == 0:
            # No NDVI rows for this field×year (after grouping)
            features.append({
                "field_id": field_id,
                "year": year,
                "ndvi_observations": 0,
                "ndvi_mean_season": np.nan,
                "ndvi_max_season": np.nan,
                "ndvi_early": np.nan,
                "ndvi_mid": np.nan,
                "ndvi_late": np.nan,
            })
            continue
        
        # Calendar year filter first
        field_ndvi = field_ndvi[field_ndvi['date'].dt.year == year].copy()

        # Crop-window filter by months
        # We resolve season window from crop_id via crops.csv mapping (standard_name).
        # If crop_id missing/unknown -> default window.
        # Build mapping lazily once from crops.csv on first use.
        nonlocal_crop_map = getattr(build_ndvi_features, "_crop_id_to_std", None)
        if nonlocal_crop_map is None:
            try:
                _crops_df = pd.read_csv(CROPS_CSV)
                nonlocal_crop_map = _build_crop_id_to_standard_name(_crops_df)
            except Exception:
                nonlocal_crop_map = {}
            setattr(build_ndvi_features, "_crop_id_to_std", nonlocal_crop_map)

        start_m, end_m, _ = _season_params_for_crop_id(crop_id_val, nonlocal_crop_map)
        field_ndvi = field_ndvi[(field_ndvi["date"].dt.month >= start_m) & (field_ndvi["date"].dt.month <= end_m)].copy()

        # Сортируем по дате
        field_ndvi = field_ndvi.sort_values('date')
        
        if len(field_ndvi) == 0:
            features.append({
                "field_id": field_id,
                "year": year,
                "ndvi_observations": 0,
                "ndvi_mean_season": np.nan,
                "ndvi_max_season": np.nan,
                "ndvi_early": np.nan,
                "ndvi_mid": np.nan,
                "ndvi_late": np.nan,
            })
            continue
        
        ndvi_vals = field_ndvi['ndvi_mean'].dropna()
        
        if len(ndvi_vals) == 0:
            features.append({
                "field_id": field_id,
                "year": year,
                "ndvi_observations": 0,
                "ndvi_mean_season": np.nan,
                "ndvi_max_season": np.nan,
                "ndvi_early": np.nan,
                "ndvi_mid": np.nan,
                "ndvi_late": np.nan,
            })
            continue
        
        # Crop-window season stats
        feature_row = {
            'field_id': field_id,
            'year': year,
            'ndvi_observations': len(ndvi_vals),
            'ndvi_mean_season': ndvi_vals.mean(),
            'ndvi_max_season': ndvi_vals.max(),
        }
        
        # Season thirds by observation order (np.array_split)
        field_ndvi_sorted = field_ndvi.sort_values("date")
        if len(field_ndvi_sorted) >= 3:
            splits = np.array_split(field_ndvi_sorted["ndvi_mean"].to_numpy(dtype=float), 3)
            early_vals = pd.Series(splits[0]).dropna()
            mid_vals = pd.Series(splits[1]).dropna()
            late_vals = pd.Series(splits[2]).dropna()
            feature_row["ndvi_early"] = float(early_vals.mean()) if len(early_vals) > 0 else np.nan
            feature_row["ndvi_mid"] = float(mid_vals.mean()) if len(mid_vals) > 0 else np.nan
            feature_row["ndvi_late"] = float(late_vals.mean()) if len(late_vals) > 0 else np.nan
        else:
            feature_row["ndvi_early"] = np.nan
            feature_row["ndvi_mid"] = np.nan
            feature_row["ndvi_late"] = np.nan
        
        # We intentionally removed year-wide / redundant NDVI aggregates (trend, integral, peak date/doy, green days)
        # to keep only agronomically meaningful crop-window features requested in the prompt.
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


def merge_ndvi_with_operations_timing(
    ndvi_features: pd.DataFrame,
    operations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Добавляет фичу: дни от первой операции до максимума NDVI.
    """
    result = ndvi_features.copy()
    result['ndvi_days_first_op_to_max'] = np.nan
    
    for idx, row in result.iterrows():
        field_id = row['field_id']
        year = row['year']
        
        # Пропускаем если нет максимума NDVI
        if pd.isna(row.get('ndvi_max_season')):
            continue
        
        # Находим первую операцию сезона
        field_ops = operations_df[
            (operations_df['field_id'] == field_id) &
            (operations_df['season'] == str(year))
        ].copy()
        
        if len(field_ops) == 0:
            continue
        
        date_cols = ['planned_start_date', 'actual_start_datetime', 'completed_date']
        first_op_date = None
        
        for col in date_cols:
            if col in field_ops.columns:
                dates = pd.to_datetime(field_ops[col], errors='coerce').dropna()
                if len(dates) > 0:
                    if first_op_date is None or dates.min() < first_op_date:
                        first_op_date = dates.min()
        
        if first_op_date is None:
            continue
        
        # Находим дату максимума NDVI (нужно загрузить временной ряд)
        # Пока пропускаем, т.к. нужен доступ к ndvi_df внутри функции
        # Можно добавить позже если нужно
    
    return result


def build_targets_extended() -> pd.DataFrame:
    """
    Собирает расширенный таргет урожайности из:
    - productivity_estimates.csv (основной источник)
    - additional_yields_from_history.csv (только новые field×year)
    - yield_maps.csv (ещё новые field×year по усреднённой урожайности карты)
    
    Результат сохраняется в data_processed/targets_extended.csv с колонками:
    field_id, year, yield_t_ha
    """
    print("\n" + "=" * 80)
    print("BUILDING EXTENDED TARGETS (productivity_estimates + history + yield_maps)")
    print("=" * 80)
    print()

    if not os.path.exists(PRODUCTIVITY_CSV):
        print(f"❌ Не найден основной таргет файл: {PRODUCTIVITY_CSV}")
        return pd.DataFrame()

    # --- 1) Базовый таргет из productivity_estimates ---------------------------------
    prod_df = pd.read_csv(PRODUCTIVITY_CSV)
    base = prod_df[["field_id", "year", "estimate_value"]].copy()
    base["field_id"] = base["field_id"].astype(int)
    base["year"] = base["year"].astype(int)
    base = base.rename(columns={"estimate_value": "yield_t_ha"})

    print(f"  ✅ Базовый таргет из productivity_estimates: {len(base)} строк")

    # --- 2) Дополнительные урожайности из history_items_extended ----------------------
    if os.path.exists(ADDITIONAL_YIELDS_CSV):
        add_df = pd.read_csv(ADDITIONAL_YIELDS_CSV)
        if {"field_id", "year", "target_yield_t_ha"}.issubset(add_df.columns):
            add = add_df[["field_id", "year", "target_yield_t_ha"]].copy()
            add["field_id"] = add["field_id"].astype(int)
            add["year"] = add["year"].astype(int)
            add = add.rename(columns={"target_yield_t_ha": "yield_t_ha"})

            base_keys = set(zip(base["field_id"], base["year"]))
            mask_new = ~add.apply(lambda r: (r["field_id"], r["year"]) in base_keys, axis=1)
            add_new = add[mask_new]
            print(f"  ✅ Новых field×year из history: {len(add_new)} (из {len(add)})")
        else:
            print("  ⚠️ additional_yields_from_history.csv без нужных колонок, пропускаю дополнение")
            add_new = pd.DataFrame(columns=["field_id", "year", "yield_t_ha"])
    else:
        print("  ℹ️ additional_yields_from_history.csv не найден — используем только productivity_estimates")
        add_new = pd.DataFrame(columns=["field_id", "year", "yield_t_ha"])

    extended = pd.concat([base, add_new], ignore_index=True)

    # --- 3) Дополнительные урожайности из yield_maps ----------------------------------
    ym_new = pd.DataFrame(columns=["field_id", "year", "yield_t_ha"])
    if os.path.exists(YIELD_MAPS_CSV):
        try:
            ym_df = pd.read_csv(YIELD_MAPS_CSV)
            # Определяем год: либо готовая колонка year, либо из created_at
            if "year" not in ym_df.columns:
                if "created_at" in ym_df.columns:
                    ym_df["created_at"] = pd.to_datetime(ym_df["created_at"], errors="coerce")
                    ym_df["year"] = ym_df["created_at"].dt.year
                else:
                    print("  ⚠️ yield_maps.csv без year/created_at — не могу добавить таргеты из карт урожайности")
                    ym_df = None
            if ym_df is not None:
                # Выбираем колонку с усреднённой урожайностью
                yield_col = None
                if "calculated_average" in ym_df.columns:
                    yield_col = "calculated_average"
                elif "totals.result.average.value" in ym_df.columns:
                    yield_col = "totals.result.average.value"

                if yield_col is None:
                    print("  ⚠️ yield_maps.csv без calculated_average/totals.result.average.value — пропускаю добавление")
                elif not {"field_id", "year"}.issubset(ym_df.columns):
                    print("  ⚠️ yield_maps.csv без field_id/year — пропускаю добавление")
                else:
                    ym_sub = ym_df[["field_id", "year", yield_col]].dropna(subset=[yield_col]).copy()
                    ym_sub["field_id"] = ym_sub["field_id"].astype(int)
                    ym_sub["year"] = ym_sub["year"].astype(int)
                    ym_sub = ym_sub.rename(columns={yield_col: "yield_t_ha"})

                    # Фильтруем только те пары field×year, которых ещё нет в extended
                    ext_keys = set(zip(extended["field_id"], extended["year"]))
                    mask_new_ym = ~ym_sub.apply(lambda r: (r["field_id"], r["year"]) in ext_keys, axis=1)
                    ym_new = ym_sub[mask_new_ym]
                    print(f"  ✅ Новых field×year из yield_maps: {len(ym_new)} (из {len(ym_sub)})")
        except Exception as e:
            print(f"  ⚠️ Не удалось прочитать/обработать yield_maps.csv: {e}")

    # --- 4) Финальный extended таргет -------------------------------------------------
    targets_extended = pd.concat([extended, ym_new], ignore_index=True)
    targets_extended = targets_extended.drop_duplicates(subset=["field_id", "year"])
    targets_extended = targets_extended.sort_values(["field_id", "year"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets_extended.to_csv(TARGETS_EXTENDED_CSV, index=False)
    print(f"\n💾 targets_extended сохранён в {TARGETS_EXTENDED_CSV}")
    print(f"  Всего строк: {len(targets_extended)}")
    print()
    return targets_extended


def build_crop_history_from_history_items() -> pd.DataFrame:
    """
    Строит историю культур по годам из history_items_full.csv:
    field_id, year, crop_id
    """
    if not os.path.exists(HISTORY_ITEMS_CSV):
        print("⚠️ history_items_full.csv не найден — prev_crop_id можно будет получить только по ops/seed")
        return pd.DataFrame(columns=["field_id", "year", "crop_id"])

    hist_df = pd.read_csv(HISTORY_ITEMS_CSV)
    if not {"field_id", "year", "crop_id"}.issubset(hist_df.columns):
        print("⚠️ В history_items_full.csv нет колонок field_id/year/crop_id")
        return pd.DataFrame(columns=["field_id", "year", "crop_id"])

    tmp = hist_df.dropna(subset=["field_id", "year", "crop_id"]).copy()
    tmp["field_id"] = tmp["field_id"].astype(int)
    tmp["year"] = tmp["year"].astype(int)
    tmp["crop_id"] = tmp["crop_id"].astype(int)

    def _most_common(series: pd.Series) -> int:
        return int(Counter(series).most_common(1)[0][0])

    crop_hist = (
        tmp.groupby(["field_id", "year"])["crop_id"]
        .agg(_most_common)
        .reset_index()
    )
    print(f"  ✅ История культур из history_items: {len(crop_hist)} field×year с crop_id")
    return crop_hist


def add_crop_and_prev_crop(df: pd.DataFrame, crop_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в датасет:
    - crop_id (текущего года) из history_items (поверх того, что найдено по seeds/ops)
    - prev_crop_id: культура в предыдущем году (field_id, year-1)
    """
    if crop_hist.empty:
        # Нечего добавлять
        if "prev_crop_id" not in df.columns:
            df["prev_crop_id"] = np.nan
        return df

    out = df.copy()

    # Текущая культура
    out = out.merge(
        crop_hist.rename(columns={"crop_id": "crop_id_hist"}),
        on=["field_id", "year"],
        how="left",
    )
    if "crop_id_hist" in out.columns:
        # Если crop_id уже есть, предпочитаем hist там, где он не NaN
        if "crop_id" in out.columns:
            mask = out["crop_id_hist"].notna()
            out.loc[mask, "crop_id"] = out.loc[mask, "crop_id_hist"]
            out = out.drop(columns=["crop_id_hist"])
        else:
            out = out.rename(columns={"crop_id_hist": "crop_id"})

    # Предшественник: смещаем года +1
    prev = crop_hist.copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"crop_id": "prev_crop_id"})
    out = out.merge(prev, on=["field_id", "year"], how="left")

    return out


def build_base_extended_dataset(targets_extended: pd.DataFrame) -> pd.DataFrame:
    """
    Базовый расширенный датасет по всем field×year из targets_extended:
    field_*, ops_*, ndvi_*, crop_id, prev_crop_id, target_yield_t_ha.
    """
    print("\n" + "=" * 80)
    print("BUILDING BASE EXTENDED DATASET (field×year)")
    print("=" * 80)
    print()

    # Загрузка базовых данных
    fields_df = pd.read_csv(FIELDS_CSV)
    crops_df = pd.read_csv(CROPS_CSV)
    operations_df = pd.read_csv(OPERATIONS_CSV)
    ndvi_df = pd.read_csv(NDVI_CSV) if os.path.exists(NDVI_CSV) else None
    crop_id_to_std = _build_crop_id_to_standard_name(crops_df)

    # Seeds (опционально)
    seeds_df = None
    seeds_path = os.path.join(DATA_DIR, "seeds.csv")
    if os.path.exists(seeds_path):
        try:
            seeds_df = pd.read_csv(seeds_path)
        except Exception:
            seeds_df = None

    # Загружаем справочник удобрений (для точного NPK)
    load_fertilizers_npk()

    # Use targets_extended only as list of field×year
    base_keys = targets_extended[["field_id", "year"]].copy()
    base_keys["field_id"] = base_keys["field_id"].astype(int)
    base_keys["year"] = base_keys["year"].astype(int)

    # Add crop_id to base_keys early (needed for crop-window NDVI and weather)
    crop_hist = build_crop_history_from_history_items()
    if not crop_hist.empty:
        base_keys = base_keys.merge(crop_hist, on=["field_id", "year"], how="left", validate="one_to_one")
    else:
        base_keys["crop_id"] = np.nan

    # Создаём "фиктивный" productivity_df только с field_id/year,
    # чтобы переиспользовать create_baseline_features и build_ndvi_features
    prod_stub = base_keys.copy()
    prod_stub["estimate_value"] = 0.0  # не используется в feature-энджиниринге

    print("  🔧 Создание baseline-фич для расширенного списка field×year...")
    baseline_features = create_baseline_features(
        fields_df, crops_df, operations_df, prod_stub, seeds_df=seeds_df
    )
    print(f"    {len(baseline_features)} записей, {len(baseline_features.columns)} фич")
    
    # NDVI (crop-window) features
    if ndvi_df is not None:
        print("  🛰️ Создание NDVI-фич для расширенного списка field×year...")
        ndvi_features = build_ndvi_features(ndvi_df, prod_stub, operations_df)
        print(f"    {len(ndvi_features)} записей, {len(ndvi_features.columns) - 2} NDVI-фич (crop-window)")
    else:
        print("  ⚠️ NDVI CSV не найден, NDVI-фичи недоступны")
        ndvi_features = None
    
    df = baseline_features.copy()
    if ndvi_features is not None:
        df = df.merge(ndvi_features, on=["field_id", "year"], how="left", validate="one_to_one")

    # Soil features (по полю)
    print("  🌱 Добавление soil-фич (SoilTest)...")
    soil_feat = build_soil_features()
    if not soil_feat.empty:
        # baseline/ndvi — уникальны по (field_id, year), soil — по field_id
        df = df.merge(soil_feat, on="field_id", how="left", validate="many_to_one")
    else:
        print("  ℹ️ Soil-фичи недоступны")

    # Weather (daily, crop-window) features
    print("  🌦  Добавление weather-фич (daily, crop-window)...")
    weather_feat = build_weather_features_daily(fields_df=fields_df, keys_df=base_keys, crop_id_to_std=crop_id_to_std)
    if not weather_feat.empty:
        df = df.merge(weather_feat, on=["field_id", "year"], how="left", validate="one_to_one")
    else:
        print("  ℹ️ Weather crop-window фичи недоступны (daily history not found or invalid)")
    
    # Добавляем таргет
    df = df.merge(targets_extended, on=["field_id", "year"], how="left", validate="one_to_one")
    df = df.rename(columns={"yield_t_ha": "target_yield_t_ha"})
    
    # Add crop_id and prev_crop_id from history_items (crop_id already injected early; this also adds prev_crop_id)
    print("  🔧 Добавление crop_id и prev_crop_id из history_items...")
    df = add_crop_and_prev_crop(df, crop_hist)
    
    print(f"\n  Итоговый базовый extended датасет: {len(df)} строк, {len(df.columns)} колонок")
    return df


def build_ml_dataset_ndvi_only_extended(
    targets_extended: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> None:
    """
    Строит NDVI-only датасет на всех field×year из targets_extended:
    - field_* фичи
    - year
    - crop_id, prev_crop_id
    - все ndvi_* фичи
    Без ops_* и NPK.
    """
    print("\n" + "=" * 80)
    print("BUILDING DATASET: ml_dataset_ndvi_only_extended.csv")
    print("=" * 80)
    print()

    if targets_extended is None:
        if not os.path.exists(TARGETS_EXTENDED_CSV):
            print("❌ targets_extended.csv не найден, сначала запусти build_targets_extended()")
            return
        targets_extended = pd.read_csv(TARGETS_EXTENDED_CSV)

    base_df = build_base_extended_dataset(targets_extended)

    # Keep only the required blocks (avoid duplicate columns like field_id which also matches "field_*")
    cols_keep: list[str] = [
        "field_id",
        "year",
        "target_yield_t_ha",
        "crop_id",
        "prev_crop_id",
        "field_tillable_area",
        "field_calculated_area",
        "field_lat",
        "field_long",
    ]
    cols_keep += [c for c in base_df.columns if c.startswith("ndvi_")]
    cols_keep += [c for c in base_df.columns if c.startswith("field_soil_")]
    cols_keep += [c for c in base_df.columns if c.startswith("weather_")]
    cols_keep = list(dict.fromkeys([c for c in cols_keep if c in base_df.columns]))

    ndvi_only_df = base_df[cols_keep].copy()
    # Defensive: if upstream merges created duplicate column names, drop duplicates here.
    ndvi_only_df = ndvi_only_df.loc[:, ~ndvi_only_df.columns.duplicated()].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = output_path or ML_DATASET_NDVI_ONLY_EXT_CSV
    ndvi_only_df.to_csv(out_path, index=False)
    print(f"  💾 Сохранено: {out_path}")
    print(f"  Строк: {len(ndvi_only_df)}, колонок: {len(ndvi_only_df.columns)}")
    print()
    print(ndvi_only_df.info())
    print()


def build_ml_dataset_ops_ndvi_2021_2025(
    targets_extended: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> None:
    """
    Строит датасет с операциями+NPK+NDVI для годов, где есть операции (ops_count_total > 0).
    """
    print("\n" + "=" * 80)
    print("BUILDING DATASET: ml_dataset_ops_ndvi_2021_2025.csv")
    print("=" * 80)
    print()

    if targets_extended is None:
        if not os.path.exists(TARGETS_EXTENDED_CSV):
            print("❌ targets_extended.csv не найден, сначала запусти build_targets_extended()")
            return
        targets_extended = pd.read_csv(TARGETS_EXTENDED_CSV)

    base_df = build_base_extended_dataset(targets_extended)

    # Фильтруем только те, где реально есть операции
    if "ops_count_total" in base_df.columns:
        mask_ops = base_df["ops_count_total"] > 0
        ops_df = base_df[mask_ops].copy()
    else:
        print("⚠️ В базовом датасете нет ops_count_total, берём все строки")
        ops_df = base_df.copy()

    # На всякий случай ограничим годами, где действительно были операции (примерно 2021+)
    if "year" in ops_df.columns:
        min_year_with_ops = int(ops_df["year"].min())
        max_year_with_ops = int(ops_df["year"].max())
        print(f"  Годы с операциями в данных: {min_year_with_ops}–{max_year_with_ops}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = output_path or ML_DATASET_OPS_NDVI_EXT_CSV
    ops_df.to_csv(out_path, index=False)
    print(f"  💾 Сохранено: {out_path}")
    print(f"  Строк: {len(ops_df)}, колонок: {len(ops_df.columns)}")
    print()
    print(ops_df.info())
    print()


def build_ml_dataset_full_extended(
    targets_extended: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> None:
    """
    Строит "полный" расширенный датасет по всем field×year из targets_extended,
    с NDVI + soil + weather, без ops_* фич.

    Сохраняет в data_processed/ml_dataset_full_extended.csv.
    """
    print("\n" + "=" * 80)
    print("BUILDING DATASET: ml_dataset_full_extended.csv")
    print("=" * 80)
    print()

    if targets_extended is None:
        if not os.path.exists(TARGETS_EXTENDED_CSV):
            print("❌ targets_extended.csv не найден, сначала запусти build_targets_extended()")
            return
        targets_extended = pd.read_csv(TARGETS_EXTENDED_CSV)

    base_df = build_base_extended_dataset(targets_extended)

    # Колонки по требованиям
    cols_keep: list[str] = [
        "field_id",
        "year",
        "target_yield_t_ha",
        "crop_id",
        "prev_crop_id",
        "field_tillable_area",
        "field_calculated_area",
        "field_lat",
        "field_long",
    ]

    # NDVI crop-window features
    cols_keep += [c for c in base_df.columns if c.startswith("ndvi_")]
    # Soil-фичи
    cols_keep += [c for c in base_df.columns if c.startswith("field_soil_")]
    # Weather-фичи
    cols_keep += [c for c in base_df.columns if c.startswith("weather_")]

    # Убираем возможные дубли в списке
    cols_keep = list(dict.fromkeys(cols_keep))
    missing_cols = [c for c in cols_keep if c not in base_df.columns]
    if missing_cols:
        print(f"ℹ️ В базовом датасете отсутствуют колонки (будут пропущены): {missing_cols}")
        cols_keep = [c for c in cols_keep if c in base_df.columns]

    full_df = base_df[cols_keep].copy()
    full_df = full_df.loc[:, ~full_df.columns.duplicated()].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = output_path or ML_DATASET_FULL_EXT_CSV
    full_df.to_csv(out_path, index=False)
    print(f"  💾 Сохранено: {out_path}")
    print(f"  Строк: {len(full_df)}, колонок: {len(full_df.columns)}")
    print()
    print(full_df.info())
    print()
    print("Пример строк:")
    print(full_df.head(5).to_string())
    print()


# ============================================================================
# MAIN DATASET BUILDING (ОРИГИНАЛЬНЫЙ ДАТАСЕТ + НОВЫЕ EXTENDED)
# ============================================================================

def main():
    print("=" * 80)
    print("BUILDING ML DATASET WITH EXTENDED FEATURES")
    print("=" * 80)
    print()
    
    # Создаем выходную директорию
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Загрузка данных
    print("📂 Загрузка данных...")
    try:
        fields_df = pd.read_csv(FIELDS_CSV)
        crops_df = pd.read_csv(CROPS_CSV)
        operations_df = pd.read_csv(OPERATIONS_CSV)
        productivity_df = pd.read_csv(PRODUCTIVITY_CSV)
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        return
    
    print(f"  ✅ Загружено {len(fields_df)} полей")
    print(f"  ✅ Загружено {len(crops_df)} культур")
    print(f"  ✅ Загружено {len(operations_df)} операций")
    print(f"  ✅ Загружено {len(productivity_df)} оценок урожайности")
    print()
    
    # Seeds catalog (опционально): data_raw/seeds.csv (получаем через fetch_seeds_catalog.py)
    seeds_df = None
    seeds_path = os.path.join("data_raw", "seeds.csv")
    if os.path.exists(seeds_path):
        try:
            seeds_df = pd.read_csv(seeds_path)
            print(f"  ✅ Загружен seeds catalog: {len(seeds_df)} (из {seeds_path})")
            print()
        except Exception as e:
            print(f"  ⚠️  Не удалось прочитать {seeds_path}: {e}")
            print()
            seeds_df = None
    
    # ====================================================================
    # Задание 1 (Perplexity): диагностика application_mix_items (удобрения)
    # ====================================================================
    print("=== Примеры application_mix_items (Fertilizer) ===")

    if 'application_mix_items' in operations_df.columns:
        fert_rows = operations_df[
            operations_df['application_mix_items'].notna() &
            operations_df['application_mix_items'].astype(str).str.contains('Fertilizer', na=False)
        ].head(10)

        def _parse_mix_items(mix_str):
            if pd.isna(mix_str) or str(mix_str).strip() in ("", "[]"):
                return []
            try:
                try:
                    return json.loads(mix_str)
                except Exception:
                    return ast.literal_eval(mix_str)
            except Exception:
                return []

        any_applicable_name = False
        for i, row in fert_rows.iterrows():
            raw = row['application_mix_items']
            print(f"Row {i}:")
            print(raw)
            items = _parse_mix_items(raw)
            fert_items = [it for it in items if isinstance(it, dict) and it.get('applicable_type') == 'Fertilizer']
            if not fert_items:
                print("  (no fertilizer items parsed)")
                print()
                continue

            for j, it in enumerate(fert_items[:3]):
                # Поля, которые просили показать
                applicable_id = it.get('applicable_id')
                applicable_name = it.get('applicable_name')  # в твоих данных обычно отсутствует
                fact_rate = it.get('fact_rate')
                planned_rate = it.get('planned_rate')
                rate_unit = it.get('rate_unit')  # часто отсутствует
                unit_id = it.get('unit_id')      # есть в данных

                if applicable_name:
                    any_applicable_name = True

                print(f"  Fertilizer item #{j+1}:")
                print(f"    keys: {sorted(list(it.keys()))}")
                print(f"    applicable_id: {applicable_id}")
                print(f"    applicable_name: {applicable_name}")
                print(f"    fact_rate: {fact_rate}")
                print(f"    planned_rate: {planned_rate}")
                print(f"    rate_unit: {rate_unit}")
                print(f"    unit_id: {unit_id}")
            print()

        if not any_applicable_name:
            print("⚠️  applicable_name отсутствует в application_mix_items.")
            print("⚠️  NPK calculated using simplified 30%/10%/10% logic. Real NPK composition may differ.")
            print()
    else:
        print("⚠️  Колонка application_mix_items отсутствует в operations.csv")
        print()
    
    # Загрузим точный NPK (если есть)
    load_fertilizers_npk()
    
    # Создание базовых фич
    print("🔧 Создание базовых фич (field_*, crop_*, ops_*)...")
    baseline_features = create_baseline_features(
        fields_df, crops_df, operations_df, productivity_df, seeds_df=seeds_df
    )
    print(f"  ✅ Создано {len(baseline_features)} записей")
    print(f"  ✅ Фич: {len(baseline_features.columns)}")
    print()
    
    # NDVI фичи
    ndvi_features = None
    if os.path.exists(NDVI_CSV):
        print("🛰️  Загрузка NDVI данных...")
        try:
            ndvi_df = pd.read_csv(NDVI_CSV)
            print(f"  ✅ Загружено {len(ndvi_df)} наблюдений NDVI")
            
            print("🔧 Создание NDVI фич (ndvi_*)...")
            ndvi_features = build_ndvi_features(ndvi_df, productivity_df, operations_df)
            print(f"  ✅ Создано {len(ndvi_features)} записей")
            print(f"  ✅ NDVI фич: {len(ndvi_features.columns) - 2}")  # -2 для field_id и year
            print()
            print("📋 Пример NDVI фич:")
            print(ndvi_features.head(5).to_string())
            print()
        except Exception as e:
            print(f"  ⚠️  Ошибка при обработке NDVI: {e}")
            import traceback
            traceback.print_exc()
            print(f"  → Продолжаю без NDVI фич")
    else:
        print("⚠️  NDVI данные не найдены (ndvi_timeseries.csv)")
        print("  → Продолжаю без NDVI фич")
    print()
    
    # Объединение фич
    print("🔗 Объединение фич...")
    final_dataset = baseline_features.copy()
    
    if ndvi_features is not None:
        final_dataset = final_dataset.merge(
            ndvi_features,
            on=['field_id', 'year'],
            how='left'
        )
        print(f"  ✅ Объединено с NDVI фичами")
    
    # Добавление таргета
    final_dataset = final_dataset.merge(
        productivity_df[['field_id', 'year', 'estimate_value']],
        on=['field_id', 'year'],
        how='inner'
    )
    final_dataset.rename(columns={'estimate_value': 'target_yield_t_ha'}, inplace=True)
    print(f"  ✅ Добавлен таргет (урожайность)")
    print()

    # ====================================================================
    # Задание 3 (Perplexity): проверка пропусков + заполнение
    # ====================================================================
    print("=== Missing values check (BEFORE fill) ===")
    missing = final_dataset.isnull().sum()
    print(missing)
    print()

    missing_pct = (missing / len(final_dataset)) * 100
    key_cols = [c for c in final_dataset.columns if c.startswith('ndvi_') or c.startswith('ops_npk_')]
    if key_cols:
        critical = missing_pct[key_cols][missing_pct[key_cols] > 10].sort_values(ascending=False)
        if len(critical) > 0:
            print("⚠️ WARNING: >10% missing in key features:")
            print(critical)
            print()

    # Заполнение NDVI-фич медианой по всему датасету
    ndvi_cols = [c for c in final_dataset.columns if c.startswith('ndvi_')]
    for col in ndvi_cols:
        if final_dataset[col].isnull().any():
            median_val = final_dataset[col].median()
            final_dataset[col] = final_dataset[col].fillna(median_val)
            print(f"Filled {col} with median: {median_val}")

    # ops_npk_*: нулями
    npk_cols = [c for c in final_dataset.columns if c.startswith('ops_npk_')]
    if npk_cols:
        final_dataset[npk_cols] = final_dataset[npk_cols].fillna(0)

    # ops_days_seeding_to_harvest: медианой по культуре (crop_name), fallback на общую медиану
    if 'ops_days_seeding_to_harvest' in final_dataset.columns:
        global_med = final_dataset['ops_days_seeding_to_harvest'].median()
        final_dataset['ops_days_seeding_to_harvest'] = (
            final_dataset.groupby('crop_name', dropna=False)['ops_days_seeding_to_harvest']
            .transform(lambda x: x.fillna(x.median() if pd.notna(x.median()) else global_med))
        )
        final_dataset['ops_days_seeding_to_harvest'] = final_dataset['ops_days_seeding_to_harvest'].fillna(global_med)

    print()
    print("=== Missing values check (AFTER fill) ===")
    missing_after = final_dataset.isnull().sum()
    print(missing_after)
    print()
    
    # One-Hot по годам (year dummies) для моделей, чувствительных к распределению по годам
    print("🧩 Добавление year dummy variables...")
    year_dummies = pd.get_dummies(final_dataset['year'], prefix='year')
    final_with_year = pd.concat([final_dataset, year_dummies], axis=1)
    print(f"  ✅ Добавлены year dummy variables: {list(year_dummies.columns)}")
    print()

    # Сохранение
    print(f"💾 Сохранение датасета...")
    final_dataset.to_csv(ML_DATASET_CSV, index=False)
    print(f"  ✅ Базовый датасет: {ML_DATASET_CSV}")
    v2_path = os.path.join(OUTPUT_DIR, "ml_dataset_with_crops_and_ndvi_v2.csv")
    final_with_year.to_csv(v2_path, index=False)
    print(f"  ✅ Датасет с year dummies: {v2_path}")
    print()
    
    # Статистика
    print("=" * 80)
    print("📊 СТАТИСТИКА ДАТАСЕТА")
    print("=" * 80)
    print()
    print(f"Всего записей: {len(final_dataset)}")
    print(f"Всего фич: {len(final_dataset.columns)}")
    print()
    
    # Группировка фич по категориям
    print("📋 ГРУППИРОВКА ФИЧ ПО КАТЕГОРИЯМ:")
    print()
    
    field_features = [c for c in final_dataset.columns if c.startswith('field_')]
    crop_features = [c for c in final_dataset.columns if c.startswith('crop_')]
    ops_features = [c for c in final_dataset.columns if c.startswith('ops_')]
    ndvi_features_list = [c for c in final_dataset.columns if c.startswith('ndvi_')]
    other_features = [c for c in final_dataset.columns if c not in 
                     field_features + crop_features + ops_features + ndvi_features_list + 
                     ['field_id', 'year', 'target_yield_t_ha', 'crop_id', 'crop_name']]
    
    print(f"🌾 FIELD_* (геометрия и координаты): {len(field_features)} фич")
    for f in field_features:
        non_null = final_dataset[f].notna().sum()
        print(f"   - {f}: {non_null}/{len(final_dataset)} заполнено")
    print()
    
    print(f"🌱 CROP_* (культура, сезонность): {len(crop_features)} фич")
    for f in crop_features:
        non_null = final_dataset[f].notna().sum()
        print(f"   - {f}: {non_null}/{len(final_dataset)} заполнено")
    print()
    
    print(f"🔧 OPS_* (агрооперации): {len(ops_features)} фич")
    for f in ops_features:
        non_null = final_dataset[f].notna().sum()
        print(f"   - {f}: {non_null}/{len(final_dataset)} заполнено")
    print()
    
    print(f"🛰️  NDVI_* (спутник): {len(ndvi_features_list)} фич")
    for f in ndvi_features_list:
        non_null = final_dataset[f].notna().sum()
        print(f"   - {f}: {non_null}/{len(final_dataset)} заполнено")
    print()
    
    if other_features:
        print(f"📊 Другие фичи: {len(other_features)}")
        for f in other_features:
            non_null = final_dataset[f].notna().sum()
            print(f"   - {f}: {non_null}/{len(final_dataset)} заполнено")
        print()
    
    # .info() и .describe()
    print("=" * 80)
    print("📊 .INFO() ДАТАСЕТА")
    print("=" * 80)
    print()
    final_dataset.info()
    print()
    
    print("=" * 80)
    print("📊 .DESCRIBE() ДАТАСЕТА")
    print("=" * 80)
    print()
    print(final_dataset.describe())
    print()
    
    # Проверка пропусков в NDVI фичах
    if ndvi_features_list:
        print("=" * 80)
        print("📊 ПРОВЕРКА ПРОПУСКОВ В NDVI ФИЧАХ")
        print("=" * 80)
        print()
        ndvi_missing = final_dataset[ndvi_features_list].isna().sum()
        print(ndvi_missing[ndvi_missing > 0])
        if ndvi_missing.sum() == 0:
            print("✅ Нет пропусков в NDVI фичах!")
        else:
            print(f"⚠️  Всего пропусков: {ndvi_missing.sum()}")
            print(f"   Процент заполненности: {(1 - ndvi_missing.sum() / (len(final_dataset) * len(ndvi_features_list))) * 100:.1f}%")
        print()
    
    print("=" * 80)
    print("✅ ГОТОВ БАЗОВЫЙ ДАТАСЕТ (ml_dataset_with_ndvi)")
    print("=" * 80)
    print()

    # Build extended target (kept as a stable artifact used by both legacy and crop-window datasets)
    targets_extended = build_targets_extended()
    if not targets_extended.empty:
        print("=" * 80)
        print("BUILDING CROP-WINDOW DATASETS (versioned; legacy outputs are preserved)")
        print("=" * 80)
        build_ml_dataset_ndvi_only_extended(
            targets_extended,
            output_path=ML_DATASET_NDVI_ONLY_EXT_CROP_WINDOW_CSV,
        )
        build_ml_dataset_ops_ndvi_2021_2025(
            targets_extended,
            output_path=ML_DATASET_OPS_NDVI_EXT_CROP_WINDOW_CSV,
        )
        build_ml_dataset_full_extended(
            targets_extended,
            output_path=ML_DATASET_FULL_EXT_CROP_WINDOW_CSV,
        )

        if REBUILD_LEGACY_OUTPUTS:
            print("=" * 80)
            print("REBUILDING LEGACY DATASETS (overwriting old CSVs)")
            print("=" * 80)
            build_ml_dataset_ndvi_only_extended(targets_extended, output_path=ML_DATASET_NDVI_ONLY_EXT_CSV)
            build_ml_dataset_ops_ndvi_2021_2025(targets_extended, output_path=ML_DATASET_OPS_NDVI_EXT_CSV)
            build_ml_dataset_full_extended(targets_extended, output_path=ML_DATASET_FULL_EXT_CSV)
    else:
        print("⚠️ Не удалось построить targets_extended — extended ML датасеты не будут созданы")
    print()
    print("=" * 80)
    print("✅ ВСЕ ДАТАСЕТЫ СОБРАНЫ")
    print("=" * 80)


if __name__ == "__main__":
    main()
