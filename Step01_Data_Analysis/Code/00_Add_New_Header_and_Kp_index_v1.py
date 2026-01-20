from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 仕様（ここが重要）
# -----------------------------
BASE_COLS_OUT = ["year", "month", "day", "hour", "minute", "second", "milsecond",
                 "lat", "lon", "mlat", "mlon"]

STEP_HZ = Decimal("19.53125")
F_MAX_HZ = Decimal("20000.0")
# 19.53125 * 1024 = 20000
N_BINS = int((F_MAX_HZ / STEP_HZ).to_integral_value(rounding=ROUND_HALF_UP))  # 1024


def decimal_places(s: str) -> int:
    t = (s or "").strip().lower()
    if t == "" or t in {"nan", "none", "null"}:
        return 0
    if t[0] in "+-":
        t = t[1:]
    if "e" in t:
        mant = t.split("e", 1)[0]
        return len(mant.split(".", 1)[1]) if "." in mant else 0
    return len(t.split(".", 1)[1]) if "." in t else 0


def build_frequency_headers_1953_to_20000() -> List[str]:
    """
    周波数ヘッダ：19.53125Hz, 39.06250Hz, ... , 20000.00000Hz（計1024本）
    """
    freqs = [STEP_HZ * Decimal(i) for i in range(1, N_BINS + 1)]
    return [f"{float(f):.5f}Hz" for f in freqs]


FREQ_COLS = build_frequency_headers_1953_to_20000()
ALL_COLS_OUT = BASE_COLS_OUT + FREQ_COLS  # 11 + 1024 = 1035


def read_demeter_orbit_csv_any_header(path: Path) -> List[List[str]]:
    """
    先頭行が11列ヘッダでも、データ行が1035列でも、必ず全列を読む。
    """
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        first = next(reader, None)
        if first is None:
            return rows

        # 先頭が文字（year,month,...）ならヘッダ扱いで捨てる
        if len(first) >= 1 and first[0].strip().lower() in {"year", "yyyy"}:
            pass
        else:
            rows.append(first)

        for r in reader:
            rows.append(r)

    # 列数を 1035 に合わせる（足りなければ空を足す／多ければ切る）
    out = []
    exp = len(ALL_COLS_OUT)
    for r in rows:
        if len(r) < exp:
            r = r + [""] * (exp - len(r))
        elif len(r) > exp:
            r = r[:exp]
        out.append(r)
    return out


def parse_dt_from_sat(df: pd.DataFrame) -> pd.Series:
    ms = pd.to_numeric(df["milsecond"], errors="coerce").fillna(0).astype(np.int64)
    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["year"], errors="coerce"),
            month=pd.to_numeric(df["month"], errors="coerce"),
            day=pd.to_numeric(df["day"], errors="coerce"),
            hour=pd.to_numeric(df["hour"], errors="coerce"),
            minute=pd.to_numeric(df["minute"], errors="coerce"),
            second=pd.to_numeric(df["second"], errors="coerce"),
        ),
        errors="coerce",
    )
    return dt + pd.to_timedelta(ms, unit="ms")


def load_kp_csv(kp_path: Path) -> pd.DataFrame:
    kp_raw = pd.read_csv(kp_path, dtype=str)
    # 想定列：year month day hour minute sec milsec kp
    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(kp_raw["year"], errors="coerce"),
            month=pd.to_numeric(kp_raw["month"], errors="coerce"),
            day=pd.to_numeric(kp_raw["day"], errors="coerce"),
            hour=pd.to_numeric(kp_raw["hour"], errors="coerce"),
            minute=pd.to_numeric(kp_raw["minute"], errors="coerce"),
            second=pd.to_numeric(kp_raw["sec"], errors="coerce"),
        ),
        errors="coerce",
    ) + pd.to_timedelta(pd.to_numeric(kp_raw["milsec"], errors="coerce").fillna(0).astype(np.int64), unit="ms")

    kp = pd.DataFrame({"datetime": dt, "KpIndex": kp_raw["kp"]}).dropna(subset=["datetime"])
    kp = kp.sort_values("datetime").reset_index(drop=True)
    return kp


def fill_missing_and_format(df_str: pd.DataFrame, int_cols: List[str], float_cols: List[str]) -> pd.DataFrame:
    """
    欠損補完：同列の上下から線形補間（=平均含む）。端は最近傍で埋める。
    桁数：列ごとに「観測された小数桁の最大」に揃えて文字列化。
    """
    out = df_str.copy()

    # 数値化
    num = pd.DataFrame(index=out.index)
    for c in int_cols + float_cols:
        num[c] = pd.to_numeric(out[c].replace({"": np.nan, " ": np.nan}), errors="coerce")

    # 線形補間（両端も埋める）
    num2 = num.interpolate(method="linear", limit_direction="both")

    # 小数桁（列ごと）
    dec_max: Dict[str, int] = {}
    for c in float_cols:
        s = out[c].astype(str)
        d = s.map(decimal_places)
        mask = num[c].notna()
        dec_max[c] = int(d[mask].max()) if mask.any() else 6  # 何も無ければ6桁に逃がす

    # int列は整数化
    for c in int_cols:
        out[c] = num2[c].round(0).astype("Int64").astype(str)

    # float列は小数桁を揃えて出力
    for c in float_cols:
        places = max(dec_max[c], 0)
        out[c] = num2[c].map(lambda x: f"{x:.{places}f}" if pd.notna(x) else "")

    return out


def process_one_orbit(sat_csv: Path, kp_df: pd.DataFrame, out_csv: Path) -> None:
    rows = read_demeter_orbit_csv_any_header(sat_csv)
    if not rows:
        raise ValueError(f"空ファイル: {sat_csv}")

    df = pd.DataFrame(rows, columns=ALL_COLS_OUT)

    # 欠損補完
    int_cols = ["year", "month", "day", "hour", "minute", "second", "milsecond"]
    float_cols = ["lat", "lon", "mlat", "mlon"] + FREQ_COLS
    df_filled = fill_missing_and_format(df, int_cols=int_cols, float_cols=float_cols)

    # Kp付与（衛星時刻がKpの2行の間なら「早いKp」=直前Kp）
    df_filled["datetime"] = parse_dt_from_sat(df_filled)
    df_filled = df_filled.sort_values("datetime").reset_index(drop=True)

    merged = pd.merge_asof(
        df_filled,
        kp_df,
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.drop(columns=["datetime"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"OK: {sat_csv.name} -> {out_csv}")


def iter_csv_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    return sorted([x for x in p.rglob("*.csv") if x.is_file()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", required=True, help="C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step01_Data_Analysis\Data\For_Testing") # 衛星CSV（単体）またはフォルダ
    ap.add_argument("--kp", required=True, help="C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step01_Data_Analysis\Data\kpデータ_ALL(csv).csv") # Kp CSV
    ap.add_argument("--out", required=True, help="C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step01_Data_Analysis\Output\For_Testing") # 出力フォルダ
    args = ap.parse_args()

    sat_path = Path(args.sat)
    out_dir = Path(args.out)
    kp_df = load_kp_csv(Path(args.kp))

    for f in iter_csv_files(sat_path):
        process_one_orbit(f, kp_df, out_dir / f.name)


if __name__ == "__main__":
    main()
