# src/step3/step3_orbit_mapping.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import logging


# ============================================================
# Step3-1: 地震イベント ↔ 地震軌道（Step2出力） の紐づけ表を作る
#
# 仕様（あなたの文章に準拠）：
# - UTC
# - 地震発生前4時間以内: eq_time - 4h <= t <= eq_time
# - 震央から330 km以内を通過する軌道を対象
# - 距離<=330kmの「連続サンプル区間」のうち、最小距離点(t0)を含む区間を採用し
#   その最初/最後のdatetimeを pass_time_start/end とする
# - 1地震に複数軌道が該当する場合、closest_dis_km が最小の軌道を1本採用
# - ただし、採用した330km区間に |mlat| > 65° が1点でもあればその地震イベントは除外
#
# 出力（Step3 orbit map）：
# ["eq_id","eq_time","eq_lat","eq_lon","orbit_file","pass_time_start","pass_time_end",
#  "orbit_datetime_start","orbit_datetime_end","closest_dis_km"]
# ============================================================

STEP2_USECOLS = ["datetime", "lat", "lon", "mlat", "mlon"]


@dataclass(frozen=True)
class OrbitIndexRow:
    orbit_file: str
    path: Path
    orbit_datetime_start: pd.Timestamp
    orbit_datetime_end: pd.Timestamp


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required columns. Tried: {candidates}. Have: {list(df.columns)}")


# -----------------------------
# 距離：ヒュベニ（WGS84）
# -----------------------------
def hubeny_distance_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: float,
    lon2_deg: float,
) -> np.ndarray:
    """
    WGS84楕円体のヒュベニ公式（近距離向け）で距離[km]を返す。
    入力：度（deg）
    出力：km
    """
    # WGS84
    a = 6378137.0  # 長半径 [m]
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    latm = (lat1 + lat2) / 2.0
    sin_latm = np.sin(latm)

    w = np.sqrt(1.0 - e2 * sin_latm * sin_latm)
    m = a * (1.0 - e2) / (w**3)  # 子午線曲率半径
    n = a / w                    # 卯酉線曲率半径

    dy = dlat * m
    dx = dlon * n * np.cos(latm)

    dist_m = np.sqrt(dx * dx + dy * dy)
    return dist_m / 1000.0


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    mask=True の連続区間を (start_idx, end_idx) のリストで返す（end_idxは含む）。
    """
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)

    # 立ち上がり(0->1) と立ち下がり(1->0) を検出
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    return list(zip(starts.tolist(), ends.tolist()))


def build_orbit_index(step2_dir: Path, logger: logging.Logger) -> list[OrbitIndexRow]:
    """
    Step2出力フォルダの各CSVについて、orbit_datetime_start/end を作る。
    ファイル全体を読み込まず、datetime列だけをストリーミングで末尾まで走査する（安全寄り）。
    """
    files = sorted(step2_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No Step2 csv found: {step2_dir}")

    index_rows: list[OrbitIndexRow] = []

    for i, path in enumerate(files, start=1):
        logger.info(f"[INDEX {i}/{len(files)}] {path.name}")

        # 先頭1行で開始時刻
        head = pd.read_csv(path, usecols=["datetime"], nrows=1)
        if head.empty:
            continue
        t_start = pd.to_datetime(head.loc[0, "datetime"], errors="coerce", utc=True)
        if pd.isna(t_start):
            continue

        # 末尾はチャンクで最終datetimeを拾う
        t_end: Optional[pd.Timestamp] = None
        for chunk in pd.read_csv(path, usecols=["datetime"], chunksize=200_000):
            if chunk.empty:
                continue
            dt = pd.to_datetime(chunk["datetime"], errors="coerce", utc=True)
            dt = dt.dropna()
            if len(dt) > 0:
                t_end = dt.iloc[-1]

        if t_end is None or pd.isna(t_end):
            continue

        index_rows.append(
            OrbitIndexRow(
                orbit_file=path.name,
                path=path,
                orbit_datetime_start=t_start,
                orbit_datetime_end=t_end,
            )
        )

    logger.info(f"Orbit index built: {len(index_rows)} files indexed.")
    return index_rows


def _read_step2_window(path: Path, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
    """
    Step2 CSVを読み、t_min<=datetime<=t_max の区間だけ残す。
    最初はusecolsで最小限のみ読む（速度とメモリ節約）。
    """
    df = pd.read_csv(path, usecols=STEP2_USECOLS)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for c in ["lat", "lon", "mlat", "mlon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "lat", "lon", "mlat"])
    df = df[(df["datetime"] >= t_min) & (df["datetime"] <= t_max)].copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _choose_best_orbit_for_eq(
    eq_time: pd.Timestamp,
    eq_lat: float,
    eq_lon: float,
    orbit_index: list[OrbitIndexRow],
    lead_hours: float,
    dist_km_max: float,
    mlat_abs_max: float,
    logger: logging.Logger,
) -> Optional[dict]:
    """
    1地震イベントに対して、条件に合う軌道を探索し、
    closest_dis_km が最小の軌道1本を返す（無ければNone）。
    返すdictは出力行に対応するキーを持つ。
    """
    t_min = eq_time - pd.Timedelta(hours=lead_hours)
    t_max = eq_time

    # 時間範囲で候補を先に絞る（ファイルを無駄に開かない）
    candidates = [
        r for r in orbit_index
        if (r.orbit_datetime_end >= t_min) and (r.orbit_datetime_start <= t_max)
    ]

    if not candidates:
        return None

    best_row: Optional[dict] = None
    best_dist = np.inf

    for r in candidates:
        df = _read_step2_window(r.path, t_min, t_max)
        if df.empty:
            continue

        d_km = hubeny_distance_km(df["lat"].to_numpy(float), df["lon"].to_numpy(float), eq_lat, eq_lon)
        within = d_km <= dist_km_max
        if not np.any(within):
            continue

        # 「距離<=330km」の連続区間を列挙
        segments = _contiguous_segments(within)

        # 最小距離点（t0）は “within==True の中での最小” を採用
        idx_within = np.where(within)[0]
        idx_min = idx_within[np.argmin(d_km[idx_within])]

        # idx_min を含むセグメントを採用
        chosen_seg = None
        for s, e in segments:
            if s <= idx_min <= e:
                chosen_seg = (s, e)
                break
        if chosen_seg is None:
            # 理論上起きにくいが保険
            continue

        s, e = chosen_seg
        seg = df.iloc[s : e + 1].copy()

        # mlat除外条件：採用した330km区間に |mlat| > 65° が含まれるなら、この地震イベントは除外
        if np.any(np.abs(seg["mlat"].to_numpy(float)) > mlat_abs_max):
            # “イベント除外”仕様なので、他の軌道を見ても最終的に除外扱いにしたい。
            # ここではフラグを返して上位でイベントごと除外できるようにする。
            return {"__EXCLUDE_EVENT__": True}

        pass_start = seg["datetime"].iloc[0]
        pass_end = seg["datetime"].iloc[-1]

        closest_dis_km = float(np.min(d_km[s : e + 1]))

        # ここまで来たら「有効な軌道候補」
        if closest_dis_km < best_dist:
            best_dist = closest_dis_km
            best_row = {
                "orbit_file": r.orbit_file,
                "pass_time_start": pass_start,
                "pass_time_end": pass_end,
                "orbit_datetime_start": r.orbit_datetime_start,
                "orbit_datetime_end": r.orbit_datetime_end,
                "closest_dis_km": closest_dis_km,
            }

    return best_row


def run_step3_orbit_mapping(
    eq_catalog_path: Path,
    step2_dir: Path,
    out_csv_path: Path,
    logger: logging.Logger,
    lead_hours: float = 4.0,
    dist_km_max: float = 330.0,
    mlat_abs_max: float = 65.0,
) -> None:
    """
    Step3-1：地震-軌道紐づけ表（step3_orbit_map.csv）を作成する。

    - eq_catalog_path: 地震カタログCSV（デクラスタ済み本震）
    - step2_dir: Step2出力フォルダ（E:\\interim\\step2_normalized）
    - out_csv_path: 出力（E:\\tables\\step3_orbit_map.csv）
    """
    if not eq_catalog_path.exists():
        raise FileNotFoundError(f"EQ catalog not found: {eq_catalog_path}")
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step2 dir not found: {step2_dir}")

    _ensure_dir(out_csv_path.parent)

    # ---------- 地震カタログ読み込み ----------
    eq = pd.read_csv(eq_catalog_path)

    col_time = _pick_col(eq, ["eq_time", "time", "datetime", "origin_time", "ot"])
    col_lat = _pick_col(eq, ["eq_lat", "lat", "latitude"])
    col_lon = _pick_col(eq, ["eq_lon", "lon", "longitude"])
    col_id = None
    for cand in ["eq_id", "id", "event_id"]:
        if cand in eq.columns:
            col_id = cand
            break

    eq_time = pd.to_datetime(eq[col_time], errors="coerce", utc=True)
    eq_lat = pd.to_numeric(eq[col_lat], errors="coerce")
    eq_lon = pd.to_numeric(eq[col_lon], errors="coerce")

    eq = eq.assign(eq_time=eq_time, eq_lat=eq_lat, eq_lon=eq_lon).dropna(subset=["eq_time", "eq_lat", "eq_lon"]).copy()

    if col_id is None:
        eq["eq_id"] = np.arange(1, len(eq) + 1, dtype=int)
    else:
        eq["eq_id"] = eq[col_id]

    eq = eq.sort_values("eq_time").reset_index(drop=True)

    logger.info(f"EQ catalog loaded: {len(eq)} events")

    # ---------- 軌道インデックス ----------
    orbit_index = build_orbit_index(step2_dir, logger)

    # ---------- 紐づけ ----------
    rows: list[dict] = []
    unlinked = 0
    excluded = 0
    multi_candidate = 0  # “複数軌道が該当” の数を後で数えるため（簡易）

    for i, r in eq.iterrows():
        eid = r["eq_id"]
        et = r["eq_time"]
        elat = float(r["eq_lat"])
        elon = float(r["eq_lon"])

        # 候補軌道数（時間範囲だけ）を数える → 「複数該当」の粗い指標
        t_min = et - pd.Timedelta(hours=lead_hours)
        t_max = et
        candidates = [
            o for o in orbit_index
            if (o.orbit_datetime_end >= t_min) and (o.orbit_datetime_start <= t_max)
        ]
        if len(candidates) >= 2:
            multi_candidate += 1

        best = _choose_best_orbit_for_eq(
            eq_time=et,
            eq_lat=elat,
            eq_lon=elon,
            orbit_index=orbit_index,
            lead_hours=lead_hours,
            dist_km_max=dist_km_max,
            mlat_abs_max=mlat_abs_max,
            logger=logger,
        )

        if best is None:
            unlinked += 1
            continue

        # イベント除外フラグ（mlat条件）
        if best.get("__EXCLUDE_EVENT__"):
            excluded += 1
            continue

        rows.append(
            {
                "eq_id": eid,
                "eq_time": et,
                "eq_lat": elat,
                "eq_lon": elon,
                "orbit_file": best["orbit_file"],
                "pass_time_start": best["pass_time_start"],
                "pass_time_end": best["pass_time_end"],
                "orbit_datetime_start": best["orbit_datetime_start"],
                "orbit_datetime_end": best["orbit_datetime_end"],
                "closest_dis_km": best["closest_dis_km"],
            }
        )

        if (i + 1) % 100 == 0:
            logger.info(f"[MAP] processed {i+1}/{len(eq)}")

    out = pd.DataFrame(rows)

    # 出力の型を整える（datetimeをISO文字列にしても良いが、CSVならdatetimeのままでもOK）
    out = out.sort_values(["eq_time", "eq_id"]).reset_index(drop=True)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")

    logger.info("=== Step3 orbit mapping done ===")
    logger.info(f"linked   = {len(out)}")
    logger.info(f"unlinked = {unlinked}")
    logger.info(f"excluded(|mlat|>{mlat_abs_max} in pass segment) = {excluded}")
    logger.info(f"multi_candidate(time-overlap>=2) = {multi_candidate}")
    logger.info(f"saved: {out_csv_path}")


# （後で src/step3/main.py を作るときに、この関数を呼べばOK）
def run_step3_orbit_mapping_default_paths(cfg: dict, logger: logging.Logger) -> None:
    """
    paths.py（SSD割当）に従って、標準の入出力パスで実行するためのラッパ。
    - 地震カタログパスは cfg から読めるようにしておく（後でmain.pyで使いやすい）
    """
    from ..paths import EXTERNAL_DIR, INTERIM_DIR, TABLES_DIR

    eq_rel = cfg.get("eq", {}).get(
        "catalog_relpath",
        r"eq_catalog\eq_m4.8above_depth40kmbelow_200407-201012.csv",
    )
    eq_catalog_path = EXTERNAL_DIR / Path(eq_rel)

    step2_dirname = cfg.get("io", {}).get("step2_dirname", "step2_normalized")
    step2_dir = INTERIM_DIR / step2_dirname

    out_csv_path = TABLES_DIR / "step3_orbit_map.csv"

    lead_hours = float(cfg.get("eq", {}).get("lead_hours", 4.0))
    dist_km_max = float(cfg.get("eq", {}).get("dist_km_max", 330.0))
    mlat_abs_max = float(cfg.get("eq", {}).get("mlat_abs_max", 65.0))

    run_step3_orbit_mapping(
        eq_catalog_path=eq_catalog_path,
        step2_dir=step2_dir,
        out_csv_path=out_csv_path,
        logger=logger,
        lead_hours=lead_hours,
        dist_km_max=dist_km_max,
        mlat_abs_max=mlat_abs_max,
    )
