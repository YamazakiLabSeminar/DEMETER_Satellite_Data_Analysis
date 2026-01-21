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


def sniff_delimiter(path: Path, sample_bytes: int = 65536) -> str:
    """CSV区切り文字を自動推定（失敗したら','）"""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", " "])
        return dialect.delimiter
    except Exception:
        return ","


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
    区切り文字は自動推定する（, / ; / tab / space）
    """
    delim = sniff_delimiter(path)

    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
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


def kp_str_to_float(x) -> float:
    """
    Kpの表記を数値へ。
    例:
      "3"   -> 3.0
      "3.33"-> 3.33
      "3o"  -> 3.0
      "3+"  -> 3.33
      "3-"  -> 2.67
    """
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if s == "" or s in {"nan", "none", "null"}:
        return np.nan

    # 既に数値っぽい場合
    try:
        return float(s)
    except Exception:
        pass

    # 3分割表記（o, +, -）
    # "o" はそのまま、"+" は +1/3、"-" は -1/3
    base_part = s[:-1]
    mod = s[-1]
    try:
        base = float(base_part)
    except Exception:
        return np.nan

    if mod == "o":
        return base
    if mod == "+":
        return base + (1.0 / 3.0)
    if mod == "-":
        return base - (1.0 / 3.0)

    return np.nan


def load_kp_csv(kp_path: Path) -> pd.DataFrame:
    kp_raw = pd.read_csv(kp_path, dtype=str)

    # 念のため列名の空白を除去
    kp_raw.columns = [c.strip() for c in kp_raw.columns]

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

    # ★ここを変更：kpを数値に変換してKpIndexにする
    kp_num = kp_raw["kp"].map(kp_str_to_float)

    kp = pd.DataFrame({"datetime": dt, "KpIndex": kp_num}).dropna(subset=["datetime"])
    kp = kp.sort_values("datetime").reset_index(drop=True)

    # もしKpIndexがほぼNaNなら、ここで気づけるように警告
    nan_rate = kp["KpIndex"].isna().mean()
    if nan_rate > 0.5:
        print(f"[WARN] KpIndexのNaN率が高いです: {nan_rate:.1%}  (kp列の表記を確認してください)")

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

def attach_kp_nearest(df_sat: pd.DataFrame, kp_df: pd.DataFrame) -> pd.DataFrame:
    """
    df_sat: datetime列を持つ（衛星時刻）
    kp_df : datetime列とKpIndex列を持つ（Kp時刻）
    戻り値: df_sat に KpIndex を追加して返す
    仕様: 衛星時刻がKpの2行の間なら、より近い時刻のKpを採用。
         同距離なら「直前Kp」を採用（必要ならここを変更可）。
    """
    df_sat = df_sat.sort_values("datetime").reset_index(drop=True)
    kp_sorted = kp_df.sort_values("datetime").reset_index(drop=True).copy()

    # merge_asofで両側を取るため、Kp側の時刻列名を別名にする
    kp2 = kp_sorted.rename(columns={"datetime": "kp_datetime"})

    prev = pd.merge_asof(
        df_sat[["datetime"]],
        kp2,
        left_on="datetime",
        right_on="kp_datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    nxt = pd.merge_asof(
        df_sat[["datetime"]],
        kp2,
        left_on="datetime",
        right_on="kp_datetime",
        direction="forward",
        allow_exact_matches=True,
        suffixes=("", "_next"),
    )

    # 時刻差（秒）を計算（片側が無い場合は inf 扱い）
    prev_diff = (df_sat["datetime"] - prev["kp_datetime"]).abs()
    next_diff = (nxt["kp_datetime"] - df_sat["datetime"]).abs()

    prev_diff = prev_diff.fillna(pd.Timedelta.max)
    next_diff = next_diff.fillna(pd.Timedelta.max)

    # 近い方を採用。同距離は prev（直前）を採用（<=）
    use_prev = prev_diff <= next_diff

    kp_index = pd.Series(np.nan, index=df_sat.index, dtype="float64")
    kp_index[use_prev] = prev.loc[use_prev, "KpIndex"].astype(float)
    kp_index[~use_prev] = nxt.loc[~use_prev, "KpIndex"].astype(float)


    out = df_sat.copy()
    out["KpIndex"] = kp_index
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

    # Kp付与（近い方を採用）
    df_filled = attach_kp_nearest(df_filled, kp_df)
    merged = df_filled

    merged = merged = merged.drop(columns=["datetime"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"OK: {sat_csv.name} -> {out_csv}")


def iter_csv_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    return sorted([x for x in p.rglob("*.csv") if x.is_file()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", required=True, help="衛星CSV（単体）またはフォルダ")
    ap.add_argument("--kp", required=True, help="Kp CSV")
    ap.add_argument("--out", required=True, help="出力フォルダ")
    args = ap.parse_args()

    sat_path = Path(args.sat)
    out_dir = Path(args.out)
    kp_df = load_kp_csv(Path(args.kp))

    for f in iter_csv_files(sat_path):
        process_one_orbit(f, kp_df, out_dir / f.name)


if __name__ == "__main__":
    main()
