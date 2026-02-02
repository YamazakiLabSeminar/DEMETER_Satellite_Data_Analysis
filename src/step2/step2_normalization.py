from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# Step2: Kp merge (nearest) + binning + CDF normalization (3-pass)
#
# 入力: Step1出力CSV群（1軌道ごと）
#   必須列: datetime, lat, lon, mlat, mlon, E_1700band_mean, is_filled
#
# 入力: Kp CSV（UTC）
#   例: year,month,day,hour,minute,sec(or second),milsec(or milsecond),kp
#
# 出力: Step2出力CSV群（1軌道ごと）
#   追加列: kp_str, kp_num, kp_cat, season, mlat_bin, mlon_bin, bin_id, E_norm
# ============================================================

STEP1_REQUIRED_COLS = [
    "datetime",
    "lat",
    "lon",
    "mlat",
    "mlon",
    "E_1700band_mean",
    "is_filled",
]


@dataclass(frozen=True)
class Step2IO:
    step1_dir: Path
    kp_csv_path: Path
    out_dir: Path
    tables_dir: Path
    checkpoint_path: Path
    bin_stats_path: Path


# --------------------------
# 0) 小さいユーティリティ
# --------------------------
def _ensure_dir(p: Path) -> None:
    """フォルダが無ければ作る（あってもエラーにしない）。"""
    p.mkdir(parents=True, exist_ok=True)


def _safe_get(cfg: dict, keys: list[str], default):
    """
    cfg["a"]["b"]["c"] のようなアクセスを安全にする。
    keys=["a","b","c"] を順に辿って、無ければ default を返す。
    """
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------
# 1) Kp処理（文字列→数値→カテゴリ）
# --------------------------
def kp_str_to_num(kp: str) -> float:
    """
    例:
      '3+' -> 3 + 1/3
      '3-' -> 3 - 1/3
      '3'  -> 3
    変換できなければ NaN を返す。
    """
    if kp is None:
        return np.nan

    s = str(kp).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    try:
        if s.endswith("+"):
            return float(s[:-1]) + (1.0 / 3.0)
        if s.endswith("-"):
            return float(s[:-1]) - (1.0 / 3.0)
        return float(s)
    except Exception:
        return np.nan


def kp_num_to_cat(kp_num: float, quiet_max: float, normal_max: float) -> str:
    """
    あなたのルール（数値化→カテゴリ）：
      静か: kp_num <= quiet_max
      普通: quiet_max < kp_num < normal_max
      擾乱: kp_num >= normal_max
    """
    if np.isnan(kp_num):
        return "unknown"
    if kp_num <= quiet_max:
        return "quiet"
    if kp_num < normal_max:
        return "normal"
    return "disturb"


def load_kp_table(kp_csv_path: Path, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Kp CSVを読み、merge_asof(nearest)で結合できる形に整える。
    返すDataFrameの列:
      datetime_kp, kp_str, kp_num, kp_cat
    """
    if not kp_csv_path.exists():
        raise FileNotFoundError(f"Kp file not found: {kp_csv_path}")

    kp = pd.read_csv(kp_csv_path)

    # Kp CSVの列名揺れに少し対応（sec/second, milsec/milsecondなど）
    def pick_col(candidates: list[str]) -> str:
        for c in candidates:
            if c in kp.columns:
                return c
        raise ValueError(f"Kp CSV missing columns: {candidates}")

    col_year = pick_col(["year", "Year"])
    col_month = pick_col(["month", "Month"])
    col_day = pick_col(["day", "Day"])
    col_hour = pick_col(["hour", "Hour"])
    col_minute = pick_col(["minute", "Minute", "min"])
    col_second = pick_col(["sec", "second", "Second"])
    col_msec = pick_col(["milsec", "milsecond", "msec", "Millisecond"])
    col_kp = pick_col(["kp", "Kp", "KP"])

    # 年月日+時刻 → datetime（UTC前提）
    dt = pd.to_datetime(
        dict(
            year=kp[col_year],
            month=kp[col_month],
            day=kp[col_day],
            hour=kp[col_hour],
            minute=kp[col_minute],
            second=kp[col_second],
        ),
        errors="coerce",
    )

    # ミリ秒（存在しない/壊れているとNaTになり得るのでcoerce）
    ms = pd.to_numeric(kp[col_msec], errors="coerce")
    dt = dt + pd.to_timedelta(ms, unit="ms")

    out = pd.DataFrame(
        {
            "datetime_kp": dt,
            "kp_str": kp[col_kp].astype(str),
        }
    )

    out["kp_num"] = out["kp_str"].map(kp_str_to_num).astype(float)

    quiet_max = float(_safe_get(cfg, ["kp", "quiet_max"], 1.0))
    normal_max = float(_safe_get(cfg, ["kp", "normal_max"], 8.0 / 3.0))  # 2.666...
    out["kp_cat"] = out["kp_num"].map(lambda v: kp_num_to_cat(v, quiet_max, normal_max))

    # merge_asofの前提：datetimeで昇順
    out = out.sort_values("datetime_kp").reset_index(drop=True)

    # 欠損行を減らす（datetimeがNaTの行は結合に使えない）
    out = out.dropna(subset=["datetime_kp"]).reset_index(drop=True)

    logger.info(f"Kp table loaded: rows={len(out)}")
    return out


def attach_kp_nearest(df: pd.DataFrame, kp_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Step1の datetime に対して、最も近いKpを付与する（nearest）。
    tolerance_hours の範囲を外れると kp が欠損する。
    """
    tol_h = int(_safe_get(cfg, ["kp", "tolerance_hours"], 2))

    left = df.sort_values("datetime").reset_index(drop=True)
    right = kp_table.sort_values("datetime_kp").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="datetime",
        right_on="datetime_kp",
        direction="nearest",
        tolerance=pd.Timedelta(hours=tol_h),
    )

    merged["kp_cat"] = merged["kp_cat"].fillna("不明")
    return merged


# --------------------------
# 2) 季節・ビン分け
# --------------------------
def month_to_season(month: int, cfg: dict) -> str:
    """月→季節（春夏秋冬）を返す。"""
    spring = set(_safe_get(cfg, ["season", "spring_months"], [3, 4, 5]))
    summer = set(_safe_get(cfg, ["season", "summer_months"], [6, 7, 8]))
    autumn = set(_safe_get(cfg, ["season", "autumn_months"], [9, 10, 11]))
    winter = set(_safe_get(cfg, ["season", "winter_months"], [12, 1, 2]))

    if month in spring:
        return "spring"
    if month in summer:
        return "summer"
    if month in autumn:
        return "autumn"
    if month in winter:
        return "winter"
    return "unknown"


def add_bins(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    mlat_bin(2deg), mlon_bin(5deg), season, bin_id を追加する。
    """
    mlat_step = int(_safe_get(cfg, ["binning", "mlat_step_deg"], 2))
    mlon_step = int(_safe_get(cfg, ["binning", "mlon_step_deg"], 5))
    to_0_360 = bool(_safe_get(cfg, ["binning", "mlon_to_0_360"], True))

    # season
    months = df["datetime"].dt.month
    df["season"] = months.map(lambda m: month_to_season(int(m), cfg) if pd.notna(m) else "不明")

    # mlat_bin: floor(mlat/step)*step
    df["mlat_bin"] = (np.floor(df["mlat"] / float(mlat_step)) * float(mlat_step)).astype("Int64")

    # mlon_bin: (必要なら0..360へ変換) -> floor(mlon/step)*step
    mlon = df["mlon"].to_numpy(dtype=float)
    if to_0_360:
        mlon = np.mod(mlon, 360.0)
    mlon_bin = np.floor(mlon / float(mlon_step)) * float(mlon_step)
    df["mlon_bin"] = pd.Series(mlon_bin).astype("Int64")

    # bin_id: 4条件を1キーに
    df["bin_id"] = (
        df["mlat_bin"].astype(str)
        + "_"
        + df["mlon_bin"].astype(str)
        + "_"
        + df["season"].astype(str)
        + "_"
        + df["kp_cat"].astype(str)
    )
    return df


# --------------------------
# 3) Step1ファイル読み込み
# --------------------------
def read_step1_csv(path: Path) -> pd.DataFrame:
    """
    Step1出力CSVを読み、必要列と型を揃える。
    """
    df = pd.read_csv(path)

    # 必須列チェック
    missing = [c for c in STEP1_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Step1 file missing columns {missing}: {path.name}")

    # datetime型に直す
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 数値型に直す（壊れていたらNaN）
    for c in ["lat", "lon", "mlat", "mlon", "E_1700band_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --------------------------
# 4) CDF近似の中核（ヒスト固定区間）
# --------------------------
def _idx_from_minmax(values: np.ndarray, mn: float, mx: float, hist_bins: int, eps_range: float) -> np.ndarray:
    """
    valuesを [mn,mx] の範囲で 0..hist_bins-1 の整数indexに落とす。
    """
    if not np.isfinite(mn) or not np.isfinite(mx):
        # min/maxが壊れている場合は全部NaN扱いに近い挙動にする
        return np.zeros(len(values), dtype=int)

    if mx <= mn:
        mx = mn + eps_range

    x = (values - mn) / (mx - mn)  # 0..1
    idx = np.floor(x * hist_bins).astype(int)
    idx[idx < 0] = 0
    idx[idx >= hist_bins] = hist_bins - 1
    return idx


def pass1_collect_stats(files: list[Path], kp_table: pd.DataFrame, cfg: dict, logger: logging.Logger) -> dict:
    """
    PASS1:
      binごとの min/max/count を集める。
    返り値:
      stats[bin_id] = {"min": float, "max": float, "count": int}
    """
    stats: dict[str, dict[str, float]] = {}

    for i, f in enumerate(files, start=1):
        logger.info(f"[PASS1 {i}/{len(files)}] {f.name}")

        df = read_step1_csv(f)
        df = attach_kp_nearest(df, kp_table, cfg)
        df = add_bins(df, cfg)

        # EがNaN、bin_idがNaNの行は使えない
        df = df.dropna(subset=["E_1700band_mean", "bin_id"])

        if len(df) == 0:
            continue

        agg = df.groupby("bin_id")["E_1700band_mean"].agg(["min", "max", "count"])

        for bin_id, row in agg.iterrows():
            mn = float(row["min"])
            mx = float(row["max"])
            ct = int(row["count"])

            if bin_id not in stats:
                stats[bin_id] = {"min": mn, "max": mx, "count": ct}
            else:
                stats[bin_id]["min"] = min(stats[bin_id]["min"], mn)
                stats[bin_id]["max"] = max(stats[bin_id]["max"], mx)
                stats[bin_id]["count"] += ct

    logger.info(f"[PASS1] bins_found={len(stats)}")
    return stats


def pass2_build_hist(files: list[Path], kp_table: pd.DataFrame, stats: dict, cfg: dict, logger: logging.Logger) -> dict:
    """
    PASS2:
      PASS1で得た min/max を使って、binごとのヒストを作る。
    返り値:
      hist[bin_id] = np.ndarray(hist_bins, dtype=int64)
    """
    hist_bins = int(_safe_get(cfg, ["cdf", "hist_bins"], 256))
    eps_range = float(_safe_get(cfg, ["cdf", "eps_range"], 1.0e-12))

    hist: dict[str, np.ndarray] = {k: np.zeros(hist_bins, dtype=np.int64) for k in stats.keys()}

    for i, f in enumerate(files, start=1):
        logger.info(f"[PASS2 {i}/{len(files)}] {f.name}")

        df = read_step1_csv(f)
        df = attach_kp_nearest(df, kp_table, cfg)
        df = add_bins(df, cfg)
        df = df.dropna(subset=["E_1700band_mean", "bin_id"])

        if len(df) == 0:
            continue

        # ファイル内はbinごとにまとめて数える（速い＆メモリ少）
        for bin_id, sub in df.groupby("bin_id"):
            if bin_id not in stats:
                continue

            values = sub["E_1700band_mean"].to_numpy(dtype=float)
            mn = stats[bin_id]["min"]
            mx = stats[bin_id]["max"]
            idx = _idx_from_minmax(values, mn, mx, hist_bins, eps_range)

            # idxの出現回数を数える → histに加算
            binc = np.bincount(idx, minlength=hist_bins).astype(np.int64)
            hist[bin_id] += binc

    logger.info(f"[PASS2] hist_bins={hist_bins}")
    return hist


def build_cdf(hist: dict, stats: dict) -> dict:
    """
    hist → CDF配列を作る。
    cdf[bin_id][k] = (hist[0] + ... + hist[k]) / count
    """
    cdf: dict[str, np.ndarray] = {}
    for bin_id, h in hist.items():
        ct = int(stats[bin_id]["count"])
        if ct <= 0:
            cdf[bin_id] = np.zeros_like(h, dtype=float)
        else:
            cdf[bin_id] = np.cumsum(h, dtype=float) / float(ct)
    return cdf


def pass3_write_outputs(
    files: list[Path],
    kp_table: pd.DataFrame,
    stats: dict,
    cdf: dict,
    io: Step2IO,
    cfg: dict,
    logger: logging.Logger,
) -> None:
    """
    PASS3:
      元データ（Step1）をもう一度読み直し、
      行ごとに E_norm = CDF_bin(E) を計算して、Step2出力に保存する。
    """
    hist_bins = int(_safe_get(cfg, ["cdf", "hist_bins"], 256))
    eps_range = float(_safe_get(cfg, ["cdf", "eps_range"], 1.0e-12))

    _ensure_dir(io.out_dir)
    _ensure_dir(io.tables_dir)

    # 途中再開（完了したファイルはスキップ）
    done: set[str] = set()
    if io.checkpoint_path.exists():
        with io.checkpoint_path.open("r", encoding="utf-8") as f:
            done = {line.strip() for line in f if line.strip()}

    for i, f in enumerate(files, start=1):
        if f.name in done:
            continue

        logger.info(f"[PASS3 {i}/{len(files)}] {f.name}")

        df = read_step1_csv(f)
        df = attach_kp_nearest(df, kp_table, cfg)
        df = add_bins(df, cfg)
        df = df.dropna(subset=["E_1700band_mean", "bin_id"])

        if len(df) == 0:
            logger.warning(f"[PASS3] no valid rows: {f.name}")
            continue

        # ---- E_normを計算（binごとにまとめて高速化） ----
        e = df["E_1700band_mean"].to_numpy(dtype=float)
        bin_ids = df["bin_id"].astype(str).to_numpy()

        e_norm = np.empty(len(df), dtype=float)

        # 「同じbin_idの行」ごとにまとめて処理する
        idx_all = np.arange(len(df))
        groups = pd.Series(idx_all).groupby(bin_ids).groups

        for bin_id, idx_rows in groups.items():
            idx_rows = np.array(list(idx_rows), dtype=int)

            if bin_id not in stats:
                e_norm[idx_rows] = np.nan
                continue

            mn = float(stats[bin_id]["min"])
            mx = float(stats[bin_id]["max"])
            idx_hist = _idx_from_minmax(e[idx_rows], mn, mx, hist_bins, eps_range)

            # CDF配列から「その区間の累積割合」を取り出す
            e_norm[idx_rows] = cdf[bin_id][idx_hist]

        df["E_norm"] = e_norm

        # ---- 出力（必要列だけ残す：容量を抑える） ----
        df_out = df[
            [
                "datetime",
                "lat",
                "lon",
                "mlat",
                "mlon",
                "kp_str",
                "kp_num",
                "kp_cat",
                "season",
                "mlat_bin",
                "mlon_bin",
                "bin_id",
                "E_1700band_mean",
                "E_norm",
                "is_filled",
            ]
        ].copy()

        out_path = io.out_dir / f"{f.stem}_step2.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8")

        # 完了したStep1ファイル名を追記（途中再開用）
        with io.checkpoint_path.open("a", encoding="utf-8") as fp:
            fp.write(f.name + "\n")

        logger.info(f"[PASS3] saved: {out_path.name} rows={len(df_out)}")


def save_bin_stats(stats: dict, out_path: Path) -> None:
    """
    PASS1で得た bin統計（count/min/max）をCSV保存する。
    """
    rows = []
    for bin_id, d in stats.items():
        parts = bin_id.split("_")
        rows.append(
            {
                "bin_id": bin_id,
                "mlat_bin": parts[0] if len(parts) > 0 else "",
                "mlon_bin": parts[1] if len(parts) > 1 else "",
                "season": parts[2] if len(parts) > 2 else "",
                "kp_cat": parts[3] if len(parts) > 3 else "",
                "count": int(d["count"]),
                "min_E": float(d["min"]),
                "max_E": float(d["max"]),
            }
        )

    df = pd.DataFrame(rows)
    _ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, encoding="utf-8")


# --------------------------
# 5) 公開API（main.pyから呼ばれる入口）
# --------------------------
def run_step2(
    step1_dir: Path,
    kp_csv_path: Path,
    out_dir: Path,
    tables_dir: Path,
    cfg: dict,
    logger: logging.Logger,
) -> None:
    """
    Step2全体を実行する入口関数。
    main.py はこの run_step2(...) だけ呼べばOK。
    """
    if not step1_dir.exists():
        raise FileNotFoundError(f"Step1 directory not found: {step1_dir}")

    files = sorted(step1_dir.glob("*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No Step1 csv found in: {step1_dir}")

    # 出力関連のパスを決める（checkpointやbin統計は tables に置く）
    checkpoint_name = str(_safe_get(cfg, ["io", "checkpoint_filename"], "checkpoint_step2_done.txt"))
    bin_stats_name = str(_safe_get(cfg, ["io", "bin_stats_filename"], "step2_bin_counts.csv"))

    io = Step2IO(
        step1_dir=step1_dir,
        kp_csv_path=kp_csv_path,
        out_dir=out_dir,
        tables_dir=tables_dir,
        checkpoint_path=tables_dir / checkpoint_name,
        bin_stats_path=tables_dir / bin_stats_name,
    )

    _ensure_dir(io.out_dir)
    _ensure_dir(io.tables_dir)

    logger.info(f"Step1 files: {len(files)}")
    logger.info(f"Step2 out_dir: {io.out_dir}")
    logger.info(f"Step2 tables_dir: {io.tables_dir}")

    # 1) Kpテーブルを作る
    kp_table = load_kp_table(io.kp_csv_path, cfg, logger)

    # 2) PASS1: min/max/count
    logger.info("=== PASS1: collect min/max/count ===")
    stats = pass1_collect_stats(files, kp_table, cfg, logger)

    # 3) bin統計を保存
    save_bin_stats(stats, io.bin_stats_path)
    logger.info(f"Saved bin stats: {io.bin_stats_path}")

    # 4) PASS2: histogram
    logger.info("=== PASS2: build histograms ===")
    hist = pass2_build_hist(files, kp_table, stats, cfg, logger)

    # 5) CDF作成
    cdf = build_cdf(hist, stats)

    # 6) PASS3: E_norm付与して保存
    logger.info("=== PASS3: write normalized outputs ===")
    pass3_write_outputs(files, kp_table, stats, cdf, io, cfg, logger)

    logger.info("Step2 finished.")
