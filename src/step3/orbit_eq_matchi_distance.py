from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#=================================================
# 内容：
# - step3_orbit_index.csv から in_eq_window == TRUE を抽出して df_orbit_index を返す
# - step3_candidate 内から対応ファイルを読み、eq_lat/eq_lon と lat/lon でヒュベニ距離を計算
# - 最小距離を min_dis として orbit_index に追加
# - 距離 <= 330km のサンプルを step3_pre_match に保存
# - step3_orbit_index.csv を更新
# 関数：
# match_orbit_distance(orbit_index_csv: Path,candidate_dir: Path,out_dir: Optional[Path] = None,)
#=================================================

def _hubeny_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """
    Hubeny distance (WGS84). Returns distance in km.
    """
    a = 6378137.0
    b = 6356752.314245
    e2 = (a**2 - b**2) / a**2

    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    avg_lat = (lat1_rad + lat2_rad) / 2.0
    sin_lat = np.sin(avg_lat)
    w = np.sqrt(1.0 - e2 * sin_lat**2)
    m = a * (1.0 - e2) / (w**3)
    n = a / w

    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad

    dist_m = np.sqrt((m * dlat) ** 2 + (n * np.cos(avg_lat) * dlon) ** 2)
    return dist_m / 1000.0


def match_orbit_distance(
    orbit_index_csv: Path,
    candidate_dir: Path,
    out_dir: Optional[Path] = None,
    out_matched_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    step3_orbit_index.csv から in_eq_window == TRUE の行を抽出し、
    各 orbit_file を step3_candidate から読み取って距離計算を行う。
    最小距離を min_dis として df_orbit_index に追加し、
    距離<=330km のサンプルを step3_pre_match に保存する。
    最後に TRUE 行のみの DataFrame を orbit_index_matched.csv に保存する。
    """
    if not orbit_index_csv.exists():
        raise FileNotFoundError(f"orbit_index CSV not found: {orbit_index_csv}")
    if not candidate_dir.exists():
        raise FileNotFoundError(f"candidate folder not found: {candidate_dir}")

    df_all = pd.read_csv(orbit_index_csv)
    required = ["orbit_file", "in_eq_window", "eq_lat", "eq_lon"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"orbit_index CSV missing columns: {missing}")

    mask = df_all["in_eq_window"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    df_orbit_index = df_all[mask].copy()

    if out_dir is None:
        out_dir = candidate_dir.parent / "step3_pre_match"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all["min_dis"] = pd.NA
    df_orbit_index["min_dis"] = pd.NA

    for idx, row in df_orbit_index.iterrows():
        orbit_file = str(row["orbit_file"])
        eq_lat = float(row["eq_lat"])
        eq_lon = float(row["eq_lon"])

        src = candidate_dir / orbit_file
        if not src.exists():
            continue

        df_orbit = pd.read_csv(src)
        if "lat" not in df_orbit.columns or "lon" not in df_orbit.columns:
            continue

        lat = pd.to_numeric(df_orbit["lat"], errors="coerce")
        lon = pd.to_numeric(df_orbit["lon"], errors="coerce")
        valid = ~(lat.isna() | lon.isna())
        if not valid.any():
            continue

        dist_km = _hubeny_distance_km(
            lat[valid].to_numpy(),
            lon[valid].to_numpy(),
            eq_lat,
            eq_lon,
        )

        df_orbit_valid = df_orbit.loc[valid].copy()
        df_orbit_valid["distance_km"] = dist_km
        df_close = df_orbit_valid[df_orbit_valid["distance_km"] <= 330.0]

        if not df_close.empty:
            min_dis = float(df_close["distance_km"].min())
            df_close.to_csv(out_dir / orbit_file, index=False, encoding="utf-8")
        else:
            min_dis = math.nan

        df_all.at[idx, "min_dis"] = min_dis
        df_orbit_index.at[idx, "min_dis"] = min_dis

    if out_matched_csv is None:
        out_matched_csv = orbit_index_csv.parent / "orbit_index_matched.csv"
    out_matched_csv.parent.mkdir(parents=True, exist_ok=True)
    df_orbit_index.to_csv(out_matched_csv, index=False, encoding="utf-8")
    return df_orbit_index
