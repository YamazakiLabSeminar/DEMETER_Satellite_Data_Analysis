from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
#=============================================
# - orbit_indexファイルを読み, データフレームを出し、TRUE/FLASE列を追加する。
# - orbit_indexファイルにおける各行の"orbit_start_imte"、"orbit_end_time"が地震ファイルの"4hour_before"から"datetime"までの間に両方入るかないかを確認し走査する。
# - 両方入ったらTRUEに判定し、マッチした最初地震の"eq_id"を付与してoribit_indexデータフレームに新しい行列として追加する。
# - なかったらFALSEとeq_idは空。
# - 元のorbit_indexファイルを今のデータフレームで更新する。
#========================================================

def match_orbit_to_eq_time(
    orbit_index_csv: Path,
    eq_csv: Path,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    orbit_index を読み込み、各行の軌道開始/終了時刻が
    地震の [4hour_before, datetime] 区間に入るか判定して
    TRUE/FALSE 列と eq_id 列を追加する。
    その後、元の orbit_index CSV を更新保存する。
    """
    if not orbit_index_csv.exists():
        raise FileNotFoundError(f"orbit_index CSV not found: {orbit_index_csv}")
    if not eq_csv.exists():
        raise FileNotFoundError(f"earthquake CSV not found: {eq_csv}")

    orbit_df = pd.read_csv(orbit_index_csv)
    eq_df = pd.read_csv(eq_csv)

    required_orbit = ["orbit_file", "orbit_start_time", "orbit_end_time"]
    missing_orbit = [c for c in required_orbit if c not in orbit_df.columns]
    if missing_orbit:
        raise ValueError(f"orbit_index CSV missing columns: {missing_orbit}")

    required_eq = ["eq_id", "4hour_before", "datetime"]
    missing_eq = [c for c in required_eq if c not in eq_df.columns]
    if missing_eq:
        raise ValueError(f"earthquake CSV missing columns: {missing_eq}")

    orbit_df = orbit_df.copy()
    eq_df = eq_df.copy()

    orbit_df["orbit_start_time"] = pd.to_datetime(
        orbit_df["orbit_start_time"], errors="coerce", format="mixed"
    )
    orbit_df["orbit_end_time"] = pd.to_datetime(
        orbit_df["orbit_end_time"], errors="coerce", format="mixed"
    )

    eq_df["4hour_before"] = pd.to_datetime(eq_df["4hour_before"], errors="coerce", format="mixed")
    eq_df["datetime"] = pd.to_datetime(eq_df["datetime"], errors="coerce", format="mixed")

    orbit_df["in_eq_window"] = False
    orbit_df["eq_id"] = pd.NA

    eq_df = eq_df.dropna(subset=["4hour_before", "datetime", "eq_id"]).copy()
    eq_df = eq_df.sort_values("datetime").reset_index(drop=True)

    for i, row in orbit_df.iterrows():
        start_t = row["orbit_start_time"]
        end_t = row["orbit_end_time"]
        if pd.isna(start_t) or pd.isna(end_t):
            continue

        mask = (eq_df["4hour_before"] <= start_t) & (end_t <= eq_df["datetime"])
        if not mask.any():
            continue

        matched = eq_df.loc[mask].iloc[0]
        orbit_df.at[i, "in_eq_window"] = True
        orbit_df.at[i, "eq_id"] = matched["eq_id"]

    if out_csv is None:
        out_csv = orbit_index_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    orbit_df.to_csv(out_csv, index=False, encoding="utf-8")

    return orbit_df
