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


def sniff_delimiter(path: Path, sample_bytes: int = 65536) -> str:
    """CSV区切り文字を自動推定（失敗したら','）
    ※スペースは列崩れを起こしやすいので候補から外す
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
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
    区切り文字は自動推定する（, / ; / tab）
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
        dec_max[c] = int(d[mask].max()) if mask.any() else 6

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
    rows = read_demeter_orbit_csv_any_header(sat_csv)
    if not rows:
        raise ValueError(f"空ファイル: {sat_csv}")

    df = pd.DataFrame(rows, columns=ALL_COLS_OUT)

    # 欠損補完
    int_cols = ["year", "month", "day", "hour", "minute", "second", "milsecond"]
    float_cols = ["lat", "lon", "mlat", "mlon"] + FREQ_COLS
    df_filled = fill_missing_and_format(df, int_cols=int_cols, float_cols=float_cols)

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
    print(f"OK: {sat_csv.name} -> {out_csv}")


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
        for f in files:
            process_one_orbit(f, kp_df, out_dir / f.name)
        return

    # 並列（任意）
    jobs = [(str(f), str(Path(args.kp)), str(out_dir / f.name)) for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        done = 0
        for fut in as_completed(futures):
            done += 1
            last = fut.result()  # 例外があればここで上がる
            if done % 50 == 0 or done == len(futures):
                print(f"[PROGRESS] {done}/{len(futures)} done (last: {Path(last).name})")


if __name__ == "__main__":
    main()
