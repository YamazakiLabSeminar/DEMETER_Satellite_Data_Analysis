from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) from lat/lon degrees."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


@dataclass(frozen=True)
class GardnerKnopoff1974Table:
    """
    Gardner & Knopoff (1974) の“窓”を、よく引用される表（M, L[km], T[days]）から補間して使う。
    値は CORSSA の review に掲載の Table 2 を採用。 :contentReference[oaicite:1]{index=1}
    """
    M: np.ndarray = field(
        default_factory=lambda: np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], dtype=float)
    )
    L_km: np.ndarray = field(
        default_factory=lambda: np.array([19.5, 22.5, 26.0, 30.0, 35.0, 40.0, 47.0, 54.0, 61.0, 70.0, 81.0, 94.0], dtype=float)
    )
    T_days: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 11.5, 22.0, 42.0, 83.0, 155.0, 290.0, 510.0, 790.0, 915.0, 960.0, 985.0], dtype=float)
    )

    def window(self, m: float) -> Tuple[float, float]:
        """
        return (L_km, T_days) for magnitude m (linear interpolation, clipped to table range)
        """
        m_clipped = float(np.clip(m, self.M.min(), self.M.max()))
        L = float(np.interp(m_clipped, self.M, self.L_km))
        T = float(np.interp(m_clipped, self.M, self.T_days))
        return L, T


def decluster_mainshocks_gk1974(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    mag_col: str = "mag",
    depth_col: Optional[str] = "depth",
    min_mag: Optional[float] = None,
    max_depth_km: Optional[float] = None,
    aftershock_only: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    GK(1974) 窓でデクラスタリングし、本震（クラスター内最大M）だけ返す。

    aftershock_only=True:
        時間窓を「本震時刻 t0 〜 t0+T」だけにして“余震除去”として動かす（おすすめ）
    False:
        「t0-T 〜 t0+T」にして foreshock も同一クラスターにまとめる

    実装方針：
    - 大きいMから順に“中心（候補本震）”として採用
    - その窓内に入る小さい地震を dependent として除外（最大Mだけ残る）
    - show_progress=True かつ tqdm が利用可能なら外側ループの進捗を表示
    """
    work = df.copy()

    # 任意フィルタ
    if min_mag is not None:
        work = work[work[mag_col] >= min_mag].copy()
    if max_depth_km is not None and depth_col and depth_col in work.columns:
        work = work[work[depth_col] <= max_depth_km].copy()

    # 時刻
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = work.dropna(subset=[time_col, lat_col, lon_col, mag_col]).copy()

    # 大きいMから（同Mなら早い時刻から）
    work = work.sort_values(by=[mag_col, time_col], ascending=[False, True]).reset_index(drop=True)

    n = len(work)
    dependent = np.zeros(n, dtype=bool)
    is_main = np.zeros(n, dtype=bool)

    table = GardnerKnopoff1974Table()

    i_iter = range(n)
    if show_progress and tqdm is not None:
        i_iter = tqdm(i_iter, total=n, desc="Declustering", unit="event")

    for i in i_iter:
        if dependent[i]:
            continue

        # i が本震（候補）として残る
        is_main[i] = True

        m0 = float(work.loc[i, mag_col])
        L_km, T_days = table.window(m0)

        t0 = work.loc[i, time_col]
        if aftershock_only:
            t_min = t0
        else:
            t_min = t0 - pd.Timedelta(days=T_days)
        t_max = t0 + pd.Timedelta(days=T_days)

        lat0 = float(work.loc[i, lat_col])
        lon0 = float(work.loc[i, lon_col])

        # i より後ろ（=より小さいM側）だけ見る
        for j in range(i + 1, n):
            if dependent[j]:
                continue

            tj = work.loc[j, time_col]
            if tj < t_min:
                continue
            if tj > t_max:
                # ※ magnitude 降順なので、時間で break できない（時刻順じゃない）
                # ここでは break しないのが安全
                continue

            lat1 = float(work.loc[j, lat_col])
            lon1 = float(work.loc[j, lon_col])
            if haversine_km(lat0, lon0, lat1, lon1) <= L_km:
                dependent[j] = True

    out = work[is_main].copy()
    out = out.sort_values(by=time_col).reset_index(drop=True)
    return out


def main():
    in_csv = r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv"
    out_csv = r"E:\tables\earthquake_catalog\declustered\eq_m4.8above_depth40kmbelow_2004-2010_declustered_ver4.csv"

    df = pd.read_csv(in_csv)

    mainshocks = decluster_mainshocks_gk1974(
        df,
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        mag_col="mag",
        depth_col="depth",
        min_mag=4.8,        # ←あなたの研究条件例
        max_depth_km=40.0,  # ←あなたの研究条件例
        aftershock_only=True,
        show_progress=True,
    )

    mainshocks.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (N={len(mainshocks)})")


if __name__ == "__main__":
    main()
