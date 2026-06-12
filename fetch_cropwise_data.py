import os
from typing import Any, Dict, List, Optional, Sequence

import requests
from requests.exceptions import HTTPError
from urllib.parse import urljoin

import pandas as pd


# ===========================
# Config
# ===========================

# 1) Вариант через переменные окружения (рекомендуется):
#    в PowerShell:
#       $env:CROPWISE_API_KEY = "...."
#       $env:CROPWISE_BASE_URL = "https://operations.cropwise.com/api/v3/"
#
# 2) Или просто впиши ключ сюда (но не коммить в git):
# !!! ТОЛЬКО САМ ТОКЕН, без "Bearer", без "X-User-Api-Token:" и т.п. !!!
# По умолчанию читаем из переменной окружения, чтобы токен не светился в коде.
API_KEY: str = os.getenv("CROPWISE_API_KEY", "PASTE_YOUR_API_KEY_HERE")

# Продовый base URL из документации Apiary:
# https://cropwiseoperations.docs.apiary.io/
BASE_URL: str = "https://operations.cropwise.com/api/v3/"

# По доке заголовок авторизации такой:
#   X-User-Api-Token: <ТОКЕН>
API_KEY_HEADER_NAME = "X-User-Api-Token"
API_KEY_HEADER_PREFIX = ""  # никакого "Bearer", "Token" и т.п., просто голый токен


# Флаги, что именно собирать.
# NB: FETCH_YIELDS / FETCH_WEATHER оставлены только чтобы не было NameError,
#     но сами эндпоинты /fields/{id}/yields и /fields/{id}/weather в API нет,
#     поэтому эти флаги ВСЕГДА должны оставаться False.
FETCH_YIELDS = False                     # НЕ ТРОГАТЬ, эндпоинта нет
FETCH_WEATHER = False                    # НЕ ТРОГАТЬ, эндпоинта нет

# Реальные эндпоинты из доки:
FETCH_OPERATIONS = True                  # /agro_operations
FETCH_CROPS = True                       # /crops
FETCH_NDVI_GRID = True                   # /ndvi_grid
FETCH_NDVI_HISTORY = False               # /historical_values?type=ndvi — временной ряд NDVI (по умолчанию OFF)
FETCH_YIELD_MAPS = True                  # /yield_maps
FETCH_PRODUCTIVITY_ESTIMATES = True      # /productivity_estimates


# ===========================
# HTTP-клиент
# ===========================

class CropwiseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        header_name: str = API_KEY_HEADER_NAME,
        header_prefix: str = API_KEY_HEADER_PREFIX,
        timeout: int = 30,
    ) -> None:
        if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
            raise ValueError("API key is empty. Задай API_KEY в конфиге.")

        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                header_name: f"{header_prefix}{api_key}",
            }
        )

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Универсальный GET с простым дебагом в случае ошибки.
        """
        url = self._url(path)
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)

        if not resp.ok:
            # Мини-дебаг, чтобы понять, почему 401/403/500 и т.п.
            print("=== HTTP ERROR ===")
            print("URL:", resp.url)
            print("STATUS:", resp.status_code)
            try:
                print("RESPONSE JSON:", resp.json())
            except Exception:
                print("RESPONSE TEXT:", resp.text)

            # Хедеры запроса (Authorization / *token* чуть замаскируем)
            req_headers = dict(resp.request.headers)
            for k in list(req_headers.keys()):
                if "auth" in k.lower() or "token" in k.lower():
                    v = req_headers[k]
                    if isinstance(v, str):
                        req_headers[k] = v[:20] + "...(hidden)"
                    else:
                        req_headers[k] = "...(hidden)"
            print("REQUEST HEADERS:", req_headers)

            resp.raise_for_status()

        return resp.json()


# ===========================
# Утилиты
# ===========================

def _as_list(obj: Any) -> List[Any]:
    """
    Привести ответ API к списку.
    Поддерживает несколько типичных форматов:
      - [ {...}, {...} ]
      - { "data": [ ... ] }
      - { "items": [ ... ] }
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("data", "items", "results"):
            if key in obj and isinstance(obj[key], Sequence):
                return list(obj[key])
        return [obj]

    return [obj]


def list_fields(client: CropwiseClient) -> List[Dict[str, Any]]:
    """Вернёт список полей (GET /fields)."""
    data = client.get("/fields")
    return _as_list(data)


def list_crops(
    client: CropwiseClient,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Список культур (crops).
    Эндпоинт из доки: /crops

    params можно использовать для фильтров из доки
    (id, name, season_type, external_id, hidden, created_at, updated_at и т.п.).
    """
    data = client.get("/crops", params=params or {})
    return _as_list(data)


def get_field_operations(
    client: CropwiseClient,
    field_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Операции на поле (посев, обработка, внесение удобрений и т.п.).

    По доке операции лежат в коллекции `agro_operations`, поэтому
    бьём туда и фильтруем по `field_id`. При необходимости можно
    добавить доп. фильтры из доки (даты, культуры и т.п.).
    """
    params: Dict[str, Any] = {"field_id": field_id}
    # если в доке есть фильтры по датам — сюда же:
    if date_from:
        params["start_date_from"] = date_from  # подправь под реальное имя поля
    if date_to:
        params["start_date_to"] = date_to      # подправь под реальное имя поля

    data = client.get("/agro_operations", params=params)
    return _as_list(data)


def get_field_ndvi_grid(
    client: CropwiseClient,
    field_id: str,
    *,
    source_sign: Optional[str] = None,
    date_eq: Optional[str] = None,
) -> Dict[str, Any]:
    """
    NDVI-решётка для поля.

    Эндпоинт из доки: GET /api/v3/ndvi_grid
      - field_id (обязательно в нашем сценарии)
      - satellite_image_id (опционально)
      - date_eq / date_gt_eq / date_lt_eq (дата снимка, YYYY-MM-DD)
      - source_sign (источник: sentinel_2, planet, и т.п.)

    Здесь используем самый частый кейс: последний доступный снимок
    для поля (если date_eq не задан) или конкретная дата.
    """
    params: Dict[str, Any] = {"field_id": field_id}
    if source_sign:
        params["source_sign"] = source_sign
    if date_eq:
        params["date_eq"] = date_eq

    raw = client.get("/ndvi_grid", params=params)

    # коллекции в Cropwise обычно отдают {"data": [...], "meta": {...}}
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return raw["data"][0] if raw["data"] else {}

    # fallback: уже объект
    if isinstance(raw, dict):
        return raw

    return {}


def get_field_ndvi_grid_stats(
    client: CropwiseClient,
    field_id: str,
    *,
    source_sign: Optional[str] = None,
    date_eq: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Агрегированные статистики NDVI по полю на один снимок:
      - ndvi_mean, ndvi_min, ndvi_max, ndvi_count.
    """
    obj = get_field_ndvi_grid(
        client,
        field_id=field_id,
        source_sign=source_sign,
        date_eq=date_eq,
    )
    if not obj:
        return {}

    points = obj.get("points")
    # в разных аккаунтах points может быть либо FeatureCollection,
    # либо просто списком фичей; поддерживаем оба варианта.
    if isinstance(points, dict):
        features = points.get("features") or []
    elif isinstance(points, list):
        features = points
    else:
        features = []

    ndvi_vals: List[float] = []
    for feat in features:
        props = feat.get("properties") or {}
        v = props.get("ndvi")
        if v is None:
            continue
        try:
            ndvi_vals.append(float(v))
        except (TypeError, ValueError):
            continue

    if not ndvi_vals:
        return {}

    stats: Dict[str, Any] = {
        "field_id": field_id,
        "date": obj.get("date") or date_eq,
        "source_sign": obj.get("source_sign") or source_sign,
        "satellite_image_id": obj.get("satellite_image_id"),
        "ndvi_mean": sum(ndvi_vals) / len(ndvi_vals),
        "ndvi_min": min(ndvi_vals),
        "ndvi_max": max(ndvi_vals),
        "ndvi_count": len(ndvi_vals),
    }
    return stats


def get_field_ndvi_history(
    client: CropwiseClient,
    field_id: str,
    *,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    source_sign: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Исторические значения NDVI по полю.

    Эндпоинт из доки: GET /api/v3a/historical_values
      required: field_id, type=ndvi
      optional: from_time, to_time, source_sign, пагинация.

    Здесь используем v3-вариант: /historical_values с теми же параметрами.
    """
    params: Dict[str, Any] = {"field_id": field_id, "type": "ndvi"}
    if from_time:
        params["from_time"] = from_time
    if to_time:
        params["to_time"] = to_time
    if source_sign:
        params["source_sign"] = source_sign

    raw = client.get("/historical_values", params=params)
    return _as_list(raw)


def get_field_yield_maps(
    client: CropwiseClient,
    field_id: str,
    *,
    season: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    source_sign: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Карты урожайности по полю.

    Эндпоинт из доки: GET /api/v3/yield_maps
      - field_id
      - season
      - date_eq / date_gt(_eq) / date_lt(_eq)
      - source_sign
    """
    params: Dict[str, Any] = {"field_id": field_id}
    if season is not None:
        params["season"] = season
    if date_from:
        params["date_gt_eq"] = date_from
    if date_to:
        params["date_lt_eq"] = date_to
    if source_sign:
        params["source_sign"] = source_sign

    raw = client.get("/yield_maps", params=params)
    return _as_list(raw)


def get_field_productivity_estimates(
    client: CropwiseClient,
    field_id: str,
    *,
    season: Optional[int] = None,
    crop_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Агрегированная продуктивность по полю/сезону.

    Эндпоинт из доки: GET /api/v3/productivity_estimates
      - field_id (обязателен в нашем сценарии)
      - season (год кампании)
      - crop_id (опционально)
    """
    params: Dict[str, Any] = {"field_id": field_id}
    if season is not None:
        params["season"] = season
    if crop_id is not None:
        params["crop_id"] = crop_id

    raw = client.get("/productivity_estimates", params=params)
    return _as_list(raw)


# ===========================
# Сбор данных в таблички
# ===========================

def export_all_to_csv(
    client: CropwiseClient,
    *,
    out_dir: str = ".",
    season: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    ndvi_from_time: Optional[str] = None,
    ndvi_to_time: Optional[str] = None,
    max_fields: Optional[int] = None,
) -> None:
    """Стягиваем всё нужное и сохраняем в CSV (только реальные эндпоинты)."""
    os.makedirs(out_dir, exist_ok=True)

    # --- поля ---
    fields = list_fields(client)
    if max_fields is not None:
        fields = fields[:max_fields]

    fields_df = pd.json_normalize(fields)
    fields_path = os.path.join(out_dir, "fields.csv")
    fields_df.to_csv(fields_path, index=False)
    print(f"Сохранил {len(fields_df)} полей -> {fields_path}")

    # --- опциональные коллекции ---
    if FETCH_CROPS:
        crops = list_crops(client)
        if crops:
            crops_df = pd.json_normalize(crops)
            crops_path = os.path.join(out_dir, "crops.csv")
            crops_df.to_csv(crops_path, index=False)
            print(f"Сохранил {len(crops_df)} культур -> {crops_path}")
        else:
            print("Нет данных по культурам (/crops вернул пусто).")

    # накопители по всем полям (per-field ресурсы)
    all_ops: List[Dict[str, Any]] = []
    ndvi_grid_stats: List[Dict[str, Any]] = []
    ndvi_history_rows: List[Dict[str, Any]] = []
    all_yield_maps: List[Dict[str, Any]] = []
    all_productivity: List[Dict[str, Any]] = []

    if (
        FETCH_OPERATIONS
        or FETCH_NDVI_GRID
        or FETCH_NDVI_HISTORY
        or FETCH_YIELD_MAPS
        or FETCH_PRODUCTIVITY_ESTIMATES
    ):
        for f in fields:
            field_id = str(f.get("id") or f.get("field_id") or f.get("uuid"))
            if not field_id:
                continue

            # урожайность
            if FETCH_YIELDS:
                try:
                    y_list = get_field_yields(client, field_id, season=season)
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"Эндпоинт урожайности для поля {field_id} не найден (404), пропускаю.")
                        y_list = []
                    else:
                        raise

                for row in y_list:
                    row = dict(row)
                    row["field_id"] = field_id
                    all_yields.append(row)

            # операции
            if FETCH_OPERATIONS:
                try:
                    ops_list = get_field_operations(
                        client,
                        field_id,
                        date_from=date_from,
                        date_to=date_to,
                    )
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"Эндпоинт операций для поля {field_id} не найден (404), пропускаю.")
                        ops_list = []
                    else:
                        raise

                for row in ops_list:
                    row = dict(row)
                    row["field_id"] = field_id
                    all_ops.append(row)

            # NDVI-решётка (один снимок на поле, агрегированные статистики)
            if FETCH_NDVI_GRID:
                try:
                    stats = get_field_ndvi_grid_stats(client, field_id)
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"Эндпоинт /ndvi_grid для поля {field_id} не найден (404), пропускаю.")
                        stats = {}
                    else:
                        raise
                if stats:
                    ndvi_grid_stats.append(stats)

            # История NDVI (временной ряд)
            if FETCH_NDVI_HISTORY:
                ndvi_items = get_field_ndvi_history(
                    client,
                    field_id,
                    from_time=ndvi_from_time,
                    to_time=ndvi_to_time,
                )
                for item in ndvi_items:
                    row = dict(item)
                    row["field_id"] = field_id
                    ndvi_history_rows.append(row)

            # Yield maps (карты урожайности)
            if FETCH_YIELD_MAPS:
                try:
                    ym_list = get_field_yield_maps(
                        client,
                        field_id,
                        season=int(season) if season is not None else None,
                        date_from=date_from,
                        date_to=date_to,
                    )
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"Эндпоинт yield_maps для поля {field_id} не найден (404), пропускаю.")
                        ym_list = []
                    else:
                        raise

                for row in ym_list:
                    row = dict(row)
                    row["field_id"] = field_id
                    all_yield_maps.append(row)

            # Productivity estimates
            if FETCH_PRODUCTIVITY_ESTIMATES:
                try:
                    pe_list = get_field_productivity_estimates(
                        client,
                        field_id,
                        season=int(season) if season is not None else None,
                    )
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"Эндпоинт productivity_estimates для поля {field_id} не найден (404), пропускаю.")
                        pe_list = []
                    else:
                        raise

                for row in pe_list:
                    row = dict(row)
                    row["field_id"] = field_id
                    all_productivity.append(row)

    # --- сохраняем остальные таблицы (если мы вообще что-то собирали) ---
    if FETCH_OPERATIONS:
        if all_ops:
            ops_df = pd.json_normalize(all_ops)
            ops_path = os.path.join(out_dir, "operations.csv")
            ops_df.to_csv(ops_path, index=False)
            print(f"Сохранил {len(ops_df)} операций -> {ops_path}")
        else:
            print("Нет данных по операциям (проверь эндпоинт/права/флаг FETCH_OPERATIONS).")

    if FETCH_NDVI_GRID:
        if ndvi_grid_stats:
            ndvi_grid_df = pd.json_normalize(ndvi_grid_stats)
            ndvi_grid_path = os.path.join(out_dir, "ndvi_grid_stats.csv")
            ndvi_grid_df.to_csv(ndvi_grid_path, index=False)
            print(f"Сохранил {len(ndvi_grid_df)} NDVI-статистик по полям -> {ndvi_grid_path}")
        else:
            print("Нет NDVI-решёток (проверь /ndvi_grid, фильтры и флаг FETCH_NDVI_GRID).")

    if FETCH_NDVI_HISTORY:
        if ndvi_history_rows:
            ndvi_hist_df = pd.json_normalize(ndvi_history_rows)
            ndvi_hist_path = os.path.join(out_dir, "ndvi_history.csv")
            ndvi_hist_df.to_csv(ndvi_hist_path, index=False)
            print(f"Сохранил {len(ndvi_hist_df)} записей NDVI-истории -> {ndvi_hist_path}")
        else:
            print("Нет NDVI-истории (проверь /historical_values?type=ndvi и флаг FETCH_NDVI_HISTORY).")

    if FETCH_YIELD_MAPS:
        if all_yield_maps:
            ym_df = pd.json_normalize(all_yield_maps)
            ym_path = os.path.join(out_dir, "yield_maps.csv")
            ym_df.to_csv(ym_path, index=False)
            print(f"Сохранил {len(ym_df)} карт урожайности -> {ym_path}")
        else:
            print("Нет данных по yield_maps (проверь эндпоинт/флаг FETCH_YIELD_MAPS).")

    if FETCH_PRODUCTIVITY_ESTIMATES:
        if all_productivity:
            pe_df = pd.json_normalize(all_productivity)
            pe_path = os.path.join(out_dir, "productivity_estimates.csv")
            pe_df.to_csv(pe_path, index=False)
            print(f"Сохранил {len(pe_df)} оценок продуктивности -> {pe_path}")
        else:
            print("Нет данных по productivity_estimates (проверь эндпоинт/флаг FETCH_PRODUCTIVITY_ESTIMATES).")


# ===========================
# Пример использования
# ===========================

def main() -> None:
    """Минимальный сценарий: стягиваем поля, операции, NDVI, yield_maps, productivity_estimates в CSV."""
    client = CropwiseClient(api_key=API_KEY)

    export_all_to_csv(
        client,
        out_dir="data_raw",
        # при необходимости можно сразу фильтрануть
        # season="2023",
        # date_from="2023-01-01",
        # date_to="2023-12-31",
        max_fields=30,  # чтобы NDVI / yield_maps не тянулись сразу по всем 100 полям
    )


if __name__ == "__main__":
    main()

