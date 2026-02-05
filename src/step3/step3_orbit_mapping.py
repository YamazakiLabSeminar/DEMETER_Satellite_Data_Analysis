from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# 1) 設定（ここは必要に応じて変更）
# ============================================================

@dataclass(frozen=True)
class Step3MappingConfig:
    # 地震と軌道の条件
    lead_hours: float = 4.0          # 地震の何時間前まで見るか（eq_time-4h <= t <= eq_time）
    max_dist_km: float = 330.0       # 震央からの最大距離（km）
    mlat_abs_limit: float = 65.0     # abs(mlat) > 65 を含む区間があれば除外

    # 入出力ファイル名
    orbit_index_name: str = "step3_orbit_index.csv"
    orbit_map_name: str = "step3_orbit_map.csv"


# ============================================================
# 2) 便利関数：CSVから先頭/末尾のdatetimeを高速に取る
#    （巨大CSVを丸ごと読むのを避ける）
# ============================================================

def _parse_datetime_str(dt_str: str) -> pd.Timestamp:
    """
    文字列の日時を pandas Timestamp に変換する。
    失敗したら例外を投げる（上位でログを出す想定）。
    """
    ts = pd.to_datetime(dt_str, errors="raise", utc=True)
    return ts


def get_first_last_datetime_csv(csv_path: Path, datetime_col: str = "datetime") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    CSVファイルの datetime 列について、最初と最後のデータ行の値を返す。
    - 先頭：ヘッダの次の最初のデータ行を読む
    - 末尾：ファイル末尾から逆走して最後のデータ行を読む
    """
    # ---- 先頭行（最初のデータ） ----
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
        cols = header.split(",")
        if datetime_col not in cols:
            raise ValueError(f"{csv_path.name}: '{datetime_col}' column not found in header.")
        dt_idx = cols.index(datetime_col)

        first_line = ""
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                first_line = line
                break
        if not first_line:
            raise ValueError(f"{csv_path.name}: no data lines found.")

        first_dt_str = first_line.split(",")[dt_idx]
        first_dt = _parse_datetime_str(first_dt_str)

    # ---- 末尾行（最後のデータ） ----
    # ファイル末尾から読んで、改行で区切って最後の非空行を取る
    with csv_path.open("rb") as fb:
        fb.seek(0, 2)  # ファイル末尾へ
        size = fb.tell()
        if size == 0:
            raise ValueError(f"{csv_path.name}: empty file.")

        block = b""
        pos = size
        # 末尾から少しずつ読み戻して、最後の改行まで拾う
        while pos > 0:
            read_size = min(4096, pos)
            pos -= read_size
            fb.seek(pos)
            block = fb.read(read_size) + block
            if b"\n" in block and block.count(b"\n") >= 2:
                break

        lines = block.splitlines()
        # 末尾から空行を飛ばす
        last_line_bytes = b""
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_line_bytes = lines[i]
                break
        if not last_line_bytes:
            raise ValueError(f"{csv_path.name}: could not find last data line.")

        last_line = last_line_bytes.decode("utf-8", errors="ignore")
        # 末尾がヘッダだけのケースを避ける（念のため）
        if last_line.startswith("datetime") or last_line.startswith(header):
            raise ValueError(f"{csv_path.name}: last line looks like header, invalid file format.")

        last_dt_str = last_line.split(",")[dt_idx]
        last_dt = _parse_datetime_str(last_dt_str)

    return first_dt, last_dt


# ============================================================
# 3) 距離計算：ヒュベニ公式（WGS84）
# ============================================================

def hubeny_distance_km(lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: float, lon2_deg: float) -> np.ndarray:
    """
    ヒュベニ公式（WGS84）で、(lat1, lon1) の各点から (lat2, lon2) までの距離[km]を返す。
    lat1_deg, lon1_deg: numpy配列（複数点）
    lat2_deg, lon2_deg: スカラー（震央）
    """
    # WGS84
    a = 6378137.0                 # 長半径 [m]
    f = 1 / 298.257223563         # 扁平率
    b = a * (1 - f)               # 短半径 [m]
    e2 = (a**2 - b**2) / a**2     # 第一離心率^2

    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    latm = (lat1 + lat2) / 2.0
    dlat = lat1 - lat2
    dlon = lon1 - lon2

    sin_latm = np.sin(latm)
    w = np.sqrt(1.0 - e2 * (sin_latm**2))
    m = a * (1.0 - e2) / (w**3)   # 子午線曲率半径 [m]
    n = a / w                     # 卯酉線曲率半径 [m]

    dy = m * dlat
    dx = n * np.cos(latm) * dlon

    dist_m = np.sqrt(dx**2 + dy**2)
    return dist_m / 1000.0


# ============================================================
# 4) 距離<=330km の「最小距離点を含む連続区間」を取る
# ============================================================

def find_segment_around_min(mask: np.ndarray, idx_min: int) -> Tuple[int, int]:
    """
    mask が True の連続区間のうち、idx_min を含む区間の [start_idx, end_idx] を返す。
    end_idx は inclusive（両端含む）。
    """
    n = len(mask)
    start = idx_min
    end = idx_min

    while start - 1 >= 0 and mask[start - 1]:
        start -= 1
    while end + 1 < n and mask[end + 1]:
        end += 1

    return start, end


# ============================================================
# 5) メイン処理：orbit_index作成 → 地震ごとに最良軌道を選ぶ
# ============================================================

def build_orbit_index(step2_dir: Path, out_csv: Path) -> pd.DataFrame:
    """
    step2_dir の全CSVについて、(orbit_file, orbit_start_time, orbit_end_time) を作り CSV 出力する。
    """
    rows = []
    csv_files = sorted(step2_dir.glob("*.csv"))
    total_files = len(csv_files)
    progress_every = max(1, total_files // 20)  # 5%刻みで表示

    for i, csv_path in enumerate(csv_files, start=1):
        first_dt, last_dt = get_first_last_datetime_csv(csv_path, datetime_col="datetime")
        rows.append(
            {
                "orbit_file": csv_path.name,
                "orbit_start_time": first_dt.isoformat(),
                "orbit_end_time": last_dt.isoformat(),
            }
        )
        if (i == 1) or (i % progress_every == 0) or (i == total_files):
            pct = (i / total_files) * 100.0 if total_files > 0 else 100.0
            print(f"[INDEX] {i}/{total_files} ({pct:.1f}%)")

    df_index = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_csv(out_csv, index=False, encoding="utf-8")
    return df_index


def run_step3_orbit_mapping(
    eq_catalog_path: Path,
    step2_dir: Path,
    tables_dir: Path,
    cfg: Optional[Step3MappingConfig] = None,
    logger=None,
) -> None:
    """
    Step3-1:
    - step3_orbit_index.csv を作成
    - step3_orbit_map.csv（地震↔軌道の紐づけ表）を作成
    """
    if cfg is None:
        cfg = Step3MappingConfig()

    tables_dir.mkdir(parents=True, exist_ok=True)
    orbit_index_path = tables_dir / cfg.orbit_index_name
    orbit_map_path = tables_dir / cfg.orbit_map_name

    # ---------- ログ関数（loggerが無ければprint） ----------
    def log_info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def log_warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"[WARN] {msg}")

    # ---------- 1) 地震カタログ読み込み ----------
    log_info(f"Read earthquake catalog: {eq_catalog_path}")
    eq_df = pd.read_csv(eq_catalog_path)

    required_eq_cols = ["latitude", "longitude", "magnitude", "datetime", "event_id"]
    missing = [c for c in required_eq_cols if c not in eq_df.columns]
    if missing:
        raise ValueError(f"eq_catalog missing columns: {missing}")

    eq_df = eq_df[required_eq_cols].copy()
    # datetime はUTCにする（文字列でもここでUTC化しておく）
    eq_df["datetime"] = pd.to_datetime(eq_df["datetime"], utc=True, errors="raise")

    # ---------- 2) orbit index を作成（または読み込み） ----------
    if orbit_index_path.exists():
        log_info(f"Load orbit index: {orbit_index_path}")
        orbit_index = pd.read_csv(orbit_index_path)
    else:
        log_info(f"Build orbit index from Step2 outputs: {step2_dir}")
        orbit_index = build_orbit_index(step2_dir=step2_dir, out_csv=orbit_index_path)

    # orbit_start/end をdatetime化（UTC）
    orbit_index["orbit_start_time"] = pd.to_datetime(
        orbit_index["orbit_start_time"], utc=True, errors="raise", format="mixed"
    )
    orbit_index["orbit_end_time"] = pd.to_datetime(
        orbit_index["orbit_end_time"], utc=True, errors="raise", format="mixed"
    )

    # ---------- 3) 地震ごとに候補軌道を絞って評価 ----------
    lead_td = timedelta(hours=cfg.lead_hours)

    out_rows: List[dict] = []
    n_no_match = 0
    n_multi_candidates = 0
    total_eq = len(eq_df)
    progress_every_eq = max(1, total_eq // 20)  # 5%刻みで表示

    def report_map_progress(i: int) -> None:
        if (i == 1) or (i % progress_every_eq == 0) or (i == total_eq):
            pct = (i / total_eq) * 100.0 if total_eq > 0 else 100.0
            log_info(
                f"[MAP] {i}/{total_eq} ({pct:.1f}%) "
                f"linked={len(out_rows)} no_link={n_no_match} multi_cand={n_multi_candidates}"
            )

    for i, (_, eq) in enumerate(eq_df.iterrows(), start=1):
        eq_id = eq["event_id"]
        eq_time = eq["datetime"]
        eq_lat = float(eq["latitude"])
        eq_lon = float(eq["longitude"])

        window_start = eq_time - lead_td
        window_end = eq_time

        # (orbit_end > window_start) and (orbit_start < window_end)
        cand = orbit_index[
            (orbit_index["orbit_end_time"] > window_start) &
            (orbit_index["orbit_start_time"] < window_end)
        ].copy()

        if cand.empty:
            n_no_match += 1
            report_map_progress(i)
            continue

        # 候補が複数ある場合のカウント（後で確認用）
        if len(cand) > 1:
            n_multi_candidates += 1

        best_row: Optional[dict] = None
        best_closest = np.inf

        for _, ob in cand.iterrows():
            orbit_file = ob["orbit_file"]
            orbit_start = ob["orbit_start_time"]
            orbit_end = ob["orbit_end_time"]

            orbit_path = step2_dir / orbit_file
            if not orbit_path.exists():
                log_warn(f"Orbit file not found: {orbit_path}")
                continue

            # 軌道データ読み込み（必要最小限の列だけ）
            try:
                df = pd.read_csv(
                    orbit_path,
                    usecols=["datetime", "lat", "lon", "mlat", "mlon"],
                )
            except Exception as e:
                log_warn(f"Failed reading {orbit_file}: {e}")
                continue

            # datetime をUTCに
            try:
                df["datetime"] = pd.to_datetime(
                    df["datetime"], utc=True, errors="raise", format="mixed"
                )
            except Exception as e:
                log_warn(f"{orbit_file}: datetime parse failed: {e}")
                continue

            # 地震の4時間窓に限定（無駄計算を減らす）
            df = df[(df["datetime"] >= window_start) & (df["datetime"] <= window_end)].copy()
            if df.empty:
                continue

            # 欠損がある行は距離計算できないので落とす（連続区間の分断にもなる）
            df = df.dropna(subset=["lat", "lon", "mlat"])
            if df.empty:
                continue

            lat_arr = df["lat"].to_numpy(dtype="float64")
            lon_arr = df["lon"].to_numpy(dtype="float64")
            mlat_arr = df["mlat"].to_numpy(dtype="float64")
            dt_arr = df["datetime"].to_numpy()

            # 距離計算（ヒュベニ）
            dist_km = hubeny_distance_km(lat_arr, lon_arr, eq_lat, eq_lon)

            # 距離<=330km の区間があるか？
            mask = dist_km <= cfg.max_dist_km
            if not np.any(mask):
                # この候補軌道は不採用 → 次の候補へ
                continue

            # 「距離<=330km 区間内」で最小距離点（idx_min）
            # 注意：mask True の部分だけで argmin を取る
            idx_true = np.where(mask)[0]
            idx_min = idx_true[np.argmin(dist_km[idx_true])]

            seg_start, seg_end = find_segment_around_min(mask, idx_min)

            pass_time_start = pd.Timestamp(dt_arr[seg_start]).isoformat()
            pass_time_end = pd.Timestamp(dt_arr[seg_end]).isoformat()

            closest_in_seg = float(np.min(dist_km[seg_start:seg_end + 1]))

            # 1地震イベントについて最小距離が最小の軌道を採用
            if closest_in_seg < best_closest:
                best_closest = closest_in_seg
                best_row = {
                    "eq_id": eq_id,
                    "eq_time": eq_time.isoformat(),
                    "eq_lat": eq_lat,
                    "eq_lon": eq_lon,
                    "orbit_file": orbit_file,
                    "orbit_start_time": orbit_start.isoformat(),
                    "orbit_end_time": orbit_end.isoformat(),
                    "pass_time_start": pass_time_start,
                    "pass_time_end": pass_time_end,
                    "closest_dis_km": closest_in_seg,
                }

        # 全候補不採用 → 紐づけなし
        if best_row is None:
            n_no_match += 1
            report_map_progress(i)
            continue

        out_rows.append(best_row)
        report_map_progress(i)

    # ---------- 4) 出力 ----------
    out_df = pd.DataFrame(out_rows)

    # 指定ヘッダ順に揃える（列欠けを防ぐ）
    columns = [
        "eq_id", "eq_time", "eq_lat", "eq_lon",
        "orbit_file", "orbit_start_time", "orbit_end_time",
        "pass_time_start", "pass_time_end",
        "closest_dis_km",
    ]
    for c in columns:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[columns]

    out_df.to_csv(orbit_map_path, index=False, encoding="utf-8")

    log_info(f"Saved orbit map: {orbit_map_path} (rows={len(out_df)})")
    log_info(f"No-link events count (rough): {n_no_match}")
    log_info(f"Events with multiple time-overlap candidates: {n_multi_candidates}")
