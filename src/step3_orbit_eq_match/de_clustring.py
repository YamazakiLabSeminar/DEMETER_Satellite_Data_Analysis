# decluster_usgs_mainshock_maxmag_direct_mincols.py
# 入力CSVから columns=["time","latitude","longitude","depth","mag"] のみ使用し、
# 本震判定は「時間窓・距離窓内で最大Mの地震だけを本震として残す（それ以外は余震として除去）」。
#
# 重要:
# - 入力の dtype 要件:
#     time: object
#     latitude/longitude/depth/mag: float64
# - 計算用に time_dt (datetime64[ns, UTC]) を内部で追加するが、
#   time列は object のまま保持する。
# - 出力:
#   1) OUT_MAIN_CSV: 本震のみ（5列 + フラグ2列）
#   2) OUT_ALL_CSV : 全イベント（5列 + フラグ2列、任意）

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 0) ここにパスを直接書く
# =========================
IN_CSV = r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv"
OUT_MAIN_CSV = r"E:\tables\earthquake_catalog\declustered\eq_m4.8above_depth40kmbelow_2004-2010_declustered_ver6.csv"
OUT_ALL_CSV = r""  # 不要なら "" にしてOK

# フィルタ条件（念のため再適用）
MIN_MAG = 4.8
MAX_DEPTH_KM = 40.0

# 出力列（time_dt は出力しない）
OUTPUT_COLS = [
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "cluster_mainshock_id",  # そのイベントが属する本震のID（orig_index）
    "is_mainshock",
]


# -------------------------
# 地球上の距離（km）：Haversine
# -------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -------------------------
# デクラスタ用の窓（指定式）
#   t: days, d: km
# -------------------------
def time_window_days(mag: float) -> float:
    """
    時間窓 t [days]:
      t = 10^(0.032*M + 2.7389)  for M >= 6.5
      t = 10^(0.5409*M - 0.547)  for M < 6.5
    """
    if mag >= 6.5:
        return 10 ** (0.032 * mag + 2.7389)
    return 10 ** (0.5409 * mag - 0.547)


def dist_window_km(mag: float) -> float:
    """
    距離窓 d [km]:
      d = 10^(0.1238*M + 0.983)
    """
    return 10 ** (0.1238 * mag + 0.983)


# -------------------------
# USGS CSV 読み込み（5列のみ + dtype固定）
# -------------------------
def load_usgs_csv_mincols(path: str | Path) -> pd.DataFrame:
    usecols = ["time", "latitude", "longitude", "depth", "mag"]

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "time": "object",  # 要求どおり object のまま
            "latitude": "float64",
            "longitude": "float64",
            "depth": "float64",
            "mag": "float64",
        },
    )

    # 計算用：datetime列（time自体はobjectのまま保持）
    # USGSの "....Z" を確実に扱うため utc=True
    df["time_dt"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # 欠損を落とす（time_dtがNaTの行も除外）
    df = df.dropna(subset=["time_dt", "latitude", "longitude", "depth", "mag"]).copy()

    return df


# -------------------------
# クラスター内最大Mのみ残すデクラスタ
# -------------------------
def decluster_keep_maxmag_mainshocks(df: pd.DataFrame) -> pd.DataFrame:
    df0 = df.sort_values("time_dt").reset_index(drop=False).rename(columns={"index": "orig_index"}).copy()
    n = len(df0)

    # ★ここ重要：numpy.datetime64 に揃える（Timestampエラー回避）
    times = (
        df0["time_dt"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    lats = df0["latitude"].to_numpy(dtype=float)
    lons = df0["longitude"].to_numpy(dtype=float)
    mags = df0["mag"].to_numpy(dtype=float)

    cluster_main_id = np.full(n, -1, dtype=int)
    is_main = np.zeros(n, dtype=bool)

    # 大きいM優先、同Mなら早い時刻優先
    order = np.lexsort((times, -mags))

    for idx in order:
        if cluster_main_id[idx] != -1:
            continue

        # idx を本震として採用
        is_main[idx] = True
        main_id = int(df0.at[idx, "orig_index"])
        cluster_main_id[idx] = main_id

        m0 = float(mags[idx])
        t_days = float(time_window_days(m0))
        d_km = float(dist_window_km(m0))

        t0 = times[idx]

        # ★余震のみ：本震後だけ（0 < Δt <= t_days）
        dt_days = (times - t0) / np.timedelta64(1, "D")
        time_mask = (dt_days > 0) & (dt_days <= t_days)

        candidates = np.where((cluster_main_id == -1) & time_mask)[0]

        lat0 = float(lats[idx])
        lon0 = float(lons[idx])

        for j in candidates:
            d = haversine_km(lat0, lon0, float(lats[j]), float(lons[j]))
            if d <= d_km:
                cluster_main_id[j] = main_id

    out = df0.copy()
    out["cluster_mainshock_id"] = cluster_main_id
    out["is_mainshock"] = is_main
    return out


def ensure_parent_dir(file_path: str) -> None:
    if not file_path:
        return
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    in_path = Path(IN_CSV)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {IN_CSV}")

    # 1) 読み込み（5列のみ）
    df = load_usgs_csv_mincols(IN_CSV)

    # 2) 条件で絞る（念のため）
    df = df[(df["mag"] >= MIN_MAG) & (df["depth"] <= MAX_DEPTH_KM)].copy()

    # 3) デクラスタ（全イベント＋フラグ）
    all_with_flags = decluster_keep_maxmag_mainshocks(df)

    # 4) 本震のみ抽出
    mainshocks = all_with_flags[all_with_flags["is_mainshock"]].copy()

    # 5) 出力列を整形（time_dt や orig_index は出力しない）
    all_out = all_with_flags[OUTPUT_COLS].copy()
    main_out = mainshocks[OUTPUT_COLS].copy()

    # 6) 保存
    ensure_parent_dir(OUT_MAIN_CSV)
    main_out.to_csv(OUT_MAIN_CSV, index=False)

    if OUT_ALL_CSV:
        ensure_parent_dir(OUT_ALL_CSV)
        all_out.to_csv(OUT_ALL_CSV, index=False)

    # 7) ログ表示
    print("=== Done ===")
    print(f"Input events (after filter): {len(df)}")
    print(f"Mainshocks:                 {len(main_out)}")
    print(f"Saved mainshocks -> {OUT_MAIN_CSV}")
    if OUT_ALL_CSV:
        print(f"Saved all+flags  -> {OUT_ALL_CSV}")


if __name__ == "__main__":
    main()

