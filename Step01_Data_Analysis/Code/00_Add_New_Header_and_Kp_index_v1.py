from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# -----------------------------
# 仕様（ここが重要）
# -----------------------------
BASE_COLS_OUT = ["year", "month", "day", "hour", "minute", "second", "milsecond",
                 "lat", "lon", "mlat", "mlon"]

STEP_HZ = Decimal("19.53125")
F_MAX_HZ = Decimal("20000.0")
N_BINS = int((F_MAX_HZ / STEP_HZ).to_integral_value(rounding=ROUND_HALF_UP))  # 1024


def detect_delimiter_fast(path: Path, sample_bytes: int = 8192) -> str:
    """
    軽量delimiter推定：最初の数KB（主に1行）から , ; \\t を数えて最大のものを選ぶ。
    ※スペースは混乱しやすいので候補に入れない。
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)

    # 先頭行中心で数える（長文でもここで十分）
    first_line = sample.splitlines()[0] if sample else ""
    counts = {
        ",": first_line.count(","),
        ";": first_line.count(";"),
        "\t": first_line.count("\t"),
    }
    # どれも0ならカンマにしておく
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def build_frequency_headers_1953_to_20000() -> List[str]:
    """
    周波数ヘッダ：19.53125Hz, 39.06250Hz, ... , 20000.00000Hz（計1024本）
    """
    freqs = [STEP_HZ * Decimal(i) for i in range(1, N_BINS + 1)]
    return [f"{float(f):.5f}Hz" for f in freqs]


FREQ_COLS = build_frequency_headers_1953_to_20000()
ALL_COLS_OUT = BASE_COLS_OUT + FREQ_COLS  # 11 + 1024 = 1035


def kp_str_to_float(x) -> float:
    """
    Kp表記を数値へ（将来の解析で使いやすい形式）
      "3"  -> 3.0
      "3o" -> 3.0
      "3+" -> 3.333...
      "3-" -> 2.666...
    """
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if s == "" or s in {"nan", "none", "null"}:
        return np.nan

    # すでに数値
    try:
        return float(s)
    except Exception:
        pass

    # 3分割表記（o, +, -）
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
    delim = detect_delimiter_fast(kp_path)
    kp_raw = pd.read_csv(kp_path, dtype=str, delimiter=delim)

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
    ) + pd.to_timedelta(
        pd.to_numeric(kp_raw["milsec"], errors="coerce").fillna(0).astype(np.int64),
        unit="ms",
    )

    kp_num = kp_raw["kp"].map(kp_str_to_float)

    kp = (
        pd.DataFrame({"datetime": dt, "KpIndex": kp_num})
        .dropna(subset=["datetime", "KpIndex"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # 軽い健全性チェック
    if kp.empty:
        print("[WARN] Kpデータが空になりました（列名や値形式を確認してください）")

    return kp


def read_satellite_csv_fast(path: Path) -> pd.DataFrame:
    """
    高速読み込み：
    - delimiterは軽量推定
    - 先頭行が year... の場合のみ1行スキップ
    - pandas C engineで読む
    - dtypeはまず float32（メモリ削減＆高速）で読む
    """
    delim = detect_delimiter_fast(path)

    # 先頭行だけ見てヘッダ判定
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        first_line = f.readline().strip().lower()

    skip = 1 if first_line.startswith("year") else 0

    # 1035列のはず。列が足りない/多い場合もあるので、pandasに任せてから補正
    df = pd.read_csv(
        path,
        delimiter=delim,
        header=None,
        names=ALL_COLS_OUT,
        skiprows=skip,
        engine="c",
        dtype=np.float32,          # まず全部 float32 で読み込む（欠損はNaN）
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False,
        on_bad_lines="skip",       # 変な行があれば飛ばす（必要なら "error" に戻してOK）
    )

    # 列数がズレた場合の保険：不足列はNaN、余りは切り捨て
    if df.shape[1] < len(ALL_COLS_OUT):
        for c in ALL_COLS_OUT[df.shape[1]:]:
            df[c] = np.nan
        df = df[ALL_COLS_OUT]
    elif df.shape[1] > len(ALL_COLS_OUT):
        df = df.iloc[:, : len(ALL_COLS_OUT)]
        df.columns = ALL_COLS_OUT

    return df


def fill_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    欠損補完（高速版）：
    - 文字列化して桁合わせをしない
    - 全列を数値のまま線形補間
    - 時刻の整数列は最後に丸める
    """
    out = df.copy()

    # まず補間（両端も埋める）
    out[ALL_COLS_OUT] = out[ALL_COLS_OUT].interpolate(method="linear", limit_direction="both")

    # 時刻の整数列だけ丸めて Int64（欠損はNaNのままでも扱える）
    int_cols = ["year", "month", "day", "hour", "minute", "second", "milsecond"]
    out[int_cols] = out[int_cols].round(0).astype("Int64")

    return out


def parse_dt_from_sat(df: pd.DataFrame) -> pd.Series:
    """
    year..milsecond から datetime を作る。
    """
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


def attach_kp_nearest_fast(df_sat: pd.DataFrame, kp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kp付与（高速版）：
    merge_asof(direction="nearest") で1回で近いKpを付与
    """
    df_sat = df_sat.sort_values("datetime").reset_index(drop=True)
    kp_df = kp_df.sort_values("datetime").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sat,
        kp_df[["datetime", "KpIndex"]],
        on="datetime",
        direction="nearest",
        allow_exact_matches=True,
        # tolerance=pd.Timedelta("6h"),  # 必要なら「近い」の上限を設定（例：6時間以上離れたらNaN）
    )
    return merged


def process_one_orbit(sat_csv: Path, kp_df: pd.DataFrame, out_csv: Path) -> None:
    # 1) 読み込み（高速）
    df = read_satellite_csv_fast(sat_csv)

    # 2) 欠損補完（高速・数値のまま）
    df_filled = fill_missing_numeric(df)

    # 3) datetime 作成 + NaT除去
    df_filled["datetime"] = parse_dt_from_sat(df_filled)
    df_filled = df_filled.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # 4) Kp付与（高速）
    df_filled = attach_kp_nearest_fast(df_filled, kp_df)

    # 5) 出力
    merged = df_filled.drop(columns=["datetime"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")

    filled_rate = merged["KpIndex"].notna().mean() if "KpIndex" in merged.columns else 0.0
    if filled_rate < 0.99:
        print(f"[WARN] {sat_csv.name}: KpIndex filled rate = {filled_rate:.2%}")
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

    files = iter_csv_files(sat_path)
    if not files:
        raise FileNotFoundError("処理対象のCSVが見つかりません。")

    for f in files:
        process_one_orbit(f, kp_df, out_dir / f.name)


if __name__ == "__main__":
    main()
