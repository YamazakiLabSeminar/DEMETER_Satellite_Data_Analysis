from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Dict

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


def sniff_delimiter(path: Path, sample_bytes: int = 4096) -> str:
    # 高速・軽量：カンマ/タブ/セミコロンを簡易判定
    with path.open("r", encoding="utf-8", errors="replace") as f:
        s = f.read(sample_bytes)
    # 行あたり複数回出てくるものを優先（閾値は雑でOK）
    c = s.count(",")
    t = s.count("\t")
    sc = s.count(";")
    if t >= c and t >= sc and t >= 5:
        return "\t"
    if sc >= c and sc >= t and sc >= 5:
        return ";"
    return ","



def build_frequency_headers_1953_to_20000() -> List[str]:
    """
    周波数ヘッダ：19.53125Hz, 39.06250Hz, ... , 20000.00000Hz（計1024本）
    """
    freqs = [STEP_HZ * Decimal(i) for i in range(1, N_BINS + 1)]
    return [f"{float(f):.5f}Hz" for f in freqs]


FREQ_COLS = build_frequency_headers_1953_to_20000()
ALL_COLS_OUT = BASE_COLS_OUT + FREQ_COLS  # 11 + 1024 = 1035


def read_demeter_orbit_as_df(path: Path) -> pd.DataFrame:
    delim = sniff_delimiter(path)

    # 1行目を見てヘッダ有無を判定
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()

    has_header = first_line.lower().startswith("year") or first_line.lower().startswith("yyyy")

    if has_header:
        df = pd.read_csv(path, sep=delim, engine="c")  # まず素直に読む（列名を活かす）
        df.columns = [str(c).strip() for c in df.columns]

        need = len(ALL_COLS_OUT)

        # 列名で不足分を補う（まとめて追加）
        missing = [c for c in ALL_COLS_OUT if c not in df.columns]
        if missing:
            df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)

        # 列順も列名で揃える（ズレ防止：ここが重要）
        df = df.reindex(columns=ALL_COLS_OUT)

        return df


    # ヘッダなし（DEMETER生データ想定）
    dtype_map = {c: "string" for c in BASE_COLS_OUT}  # 先頭11列だけ文字
    for c in FREQ_COLS:
        dtype_map[c] = "float64"  # スペクトルは最初から数値

    df = pd.read_csv(
        path,
        sep=delim,
        header=None,
        names=ALL_COLS_OUT,
        usecols=range(len(ALL_COLS_OUT)),
        engine="c",
        dtype=dtype_map,
    )

    return df



def parse_dt_from_sat(df: pd.DataFrame) -> pd.Series:
    # すべて「数値化 → 欠損補完(0) → int64」へ寄せる（NA耐性）
    y  = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype("int64")
    mo = pd.to_numeric(df["month"], errors="coerce").fillna(0).astype("int64")
    d  = pd.to_numeric(df["day"], errors="coerce").fillna(0).astype("int64")
    h  = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype("int64")
    mi = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype("int64")
    s  = pd.to_numeric(df["second"], errors="coerce").fillna(0).astype("int64")

    ms = pd.to_numeric(df["milsecond"], errors="coerce").fillna(0).astype("int64")

    dt = pd.to_datetime(
        dict(year=y, month=mo, day=d, hour=h, minute=mi, second=s),
        errors="coerce",
    )
    return dt + pd.to_timedelta(ms, unit="ms")



def kp_str_to_float(x) -> float:
    """
    Kpの表記を数値へ。
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
    kp_raw = pd.read_csv(kp_path, dtype=str, delimiter=sniff_delimiter(kp_path))

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
    ) + pd.to_timedelta(
        pd.to_numeric(kp_raw["milsec"], errors="coerce").fillna(0).astype(np.int64),
        unit="ms",
    )

    kp_num = kp_raw["kp"].map(kp_str_to_float)

    kp = pd.DataFrame({"datetime": dt, "KpIndex": kp_num}).dropna(subset=["datetime", "KpIndex"])
    kp = kp.sort_values("datetime").reset_index(drop=True)

    if kp.empty:
        print("[WARN] Kpデータが空になりました（列名や値形式を確認してください）")

    return kp


# -----------------------------
# 方針A：fragmentation回避版（ここが差分の本体）
# -----------------------------
def fill_missing_numeric_selective(
    df_str: pd.DataFrame,
    int_cols: List[str],
    pos_cols: List[str],
    spec_cols: List[str],
) -> pd.DataFrame:
    """
    要件：
    - spec_cols（周波数スペクトル列）：空白セルが無い前提 → 生データのまま数値化（補間しない）
      ※もし空白が混ざっても補間せず NaN のまま
    - int_cols（year..milsecond）：空白セルは線形補間、最終的に整数(Int64)
    - pos_cols（lat,lon,mlat,mlon）：空白セルは線形補間、小数点以下12桁にround
    """
    out = df_str.copy()

    # 1) スペクトル列：すでに read_csv で float64 読み込み済み（ここでは触らない）
    #    ※もしヘッダありのファイル（dtype=str読み）も混ざるなら、念のため float に寄せる
    if out[spec_cols].dtypes.iloc[0] == object or str(out[spec_cols].dtypes.iloc[0]).startswith("string"):
        out[spec_cols] = out[spec_cols].replace({"": np.nan, " ": np.nan}).apply(pd.to_numeric, errors="coerce")

    # 2) 時刻＋位置列：まとめて数値化→補間（fragmentation回避）
    cols_interp = int_cols + pos_cols
    num = out[cols_interp].replace({"": np.nan, " ": np.nan}).apply(pd.to_numeric, errors="coerce")

    num2 = num.interpolate(method="linear", limit_direction="both")

    # 3) int列：整数化
    for c in int_cols:
        out[c] = num2[c].round(0).astype("Int64")

    # 4) 位置列：小数12桁
    for c in pos_cols:
        out[c] = num2[c].astype(float).round(12)

    return out



def attach_kp_nearest(df_sat: pd.DataFrame, kp_df: pd.DataFrame) -> pd.DataFrame:
    """
    仕様: 衛星時刻がKpの2行の間なら、より近い時刻のKpを採用（direction="nearest" で1回）。
    """
    df_sat = df_sat.sort_values("datetime").reset_index(drop=True)
    kp_sorted = kp_df.sort_values("datetime").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sat,
        kp_sorted[["datetime", "KpIndex"]],
        on="datetime",
        direction="nearest",
        allow_exact_matches=True,
    )
    return merged


def process_one_orbit(sat_csv: Path, kp_df: pd.DataFrame, out_csv: Path) -> None:
    df = read_demeter_orbit_as_df(sat_csv)
    print("[DEBUG]", sat_csv.name, "shape=", df.shape)
    print("[DEBUG] first row base cols:", df[BASE_COLS_OUT].head(1).to_dict("records"))
    print("[DEBUG] delim=", sniff_delimiter(sat_csv))

    if df is None or df.empty:
        raise ValueError(f"空ファイル: {sat_csv}")

    # 欠損補完
    # 欠損補完（要件対応：スペクトルは補間しない）
    int_cols = ["year", "month", "day", "hour", "minute", "second", "milsecond"]
    pos_cols = ["lat", "lon", "mlat", "mlon"]
    spec_cols = FREQ_COLS

    df_filled = fill_missing_numeric_selective(
        df,
        int_cols=int_cols,
        pos_cols=pos_cols,
        spec_cols=spec_cols,
    )

    # datetime作成 + NaT除去
    df_filled["datetime"] = parse_dt_from_sat(df_filled)
    df_filled = df_filled.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Kp付与（近い方）
    df_filled = attach_kp_nearest(df_filled, kp_df)

    merged = df_filled.drop(columns=["datetime"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")

    filled_rate = merged["KpIndex"].notna().mean()
    if filled_rate < 0.99:
        print(f"[WARN] {sat_csv.name}: KpIndex filled rate = {filled_rate:.2%}")


def iter_csv_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    return sorted([x for x in p.rglob("*.csv") if x.is_file()])


# ---- 並列用（任意）----
def _worker(job: tuple[str, str, str]) -> str:
    sat_file, kp_path, out_file = job
    kp_df = load_kp_csv(Path(kp_path))  # 安定優先：各プロセスで読む
    process_one_orbit(Path(sat_file), kp_df, Path(out_file))
    return sat_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", required=True, help="入力SSD上の衛星CSV（単体）またはフォルダ")
    ap.add_argument("--kp", required=True, help="Kp CSVのパス")
    ap.add_argument("--out", required=True, help="出力SSD上の出力フォルダ")
    ap.add_argument("--workers", type=int, default=1, help="並列プロセス数（1なら逐次）")
    args = ap.parse_args()

    sat_path = Path(args.sat)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_csv_files(sat_path)
    if not files:
        raise FileNotFoundError("処理対象のCSVが見つかりません。")

    # 逐次（最も安定）
    if args.workers <= 1:
        kp_df = load_kp_csv(Path(args.kp))

        total = len(files)
        done = 0
        next_pct = 1
        print(f"[PROGRESS] 0% (0/{total})")

        for f in files:
            process_one_orbit(f, kp_df, out_dir / f.name)
            done += 1
            pct = int(done * 100 / total) if total else 100
            if pct >= next_pct:
                print(f"[PROGRESS] {pct}% ({done}/{total})")
                next_pct = pct + 1
        return

    # 並列（任意）
    jobs = [(str(f), str(Path(args.kp)), str(out_dir / f.name)) for f in files]

    total = len(jobs)
    done = 0
    next_pct = 1
    print(f"[PROGRESS] 0% (0/{total})")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]

        for fut in as_completed(futures):
            _ = fut.result()  # 例外があればここで止まる（安全）
            done += 1
            pct = int(done * 100 / total) if total else 100
            if pct >= next_pct:
                print(f"[PROGRESS] {pct}% ({done}/{total})")
                next_pct = pct + 1


if __name__ == "__main__":
    main()
