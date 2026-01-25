from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
import numpy as np


# -----------------------------
# ここは「固定仕様」：メタ列名（11列）
# -----------------------------
META_COLS = [
    "year", "month", "day", "hour", "minute", "second", "milsecond",
    "lat", "lon", "mlat", "mlon",
]


def make_freq_cols() -> list[str]:
    """
    19.53125Hz刻みで 19.53125〜20000.0 の 1024本の周波数列名を作る。
    列名は後で選びやすいように "f_XXXX.XXXXX" 形式に統一する。
    """
    step = 19.53125
    freqs = [step * i for i in range(1, 1024 + 1)]  # 1..1024
    # 小数表現を固定（5桁）してブレを防ぐ
    return [f"f_{f:.5f}" for f in freqs]


def band_freq_cols(f_low: float, f_high: float) -> list[str]:
    """
    指定帯域 [f_low, f_high] に含まれる周波数列名を返す。
    """
    step = 19.53125
    freqs = np.array([step * i for i in range(1, 1024 + 1)], dtype=float)
    mask = (freqs >= f_low) & (freqs <= f_high)
    cols = [f"f_{f:.5f}" for f in freqs[mask]]
    return cols


def read_demeter_csv_as_1035cols(csv_path: Path) -> pd.DataFrame:
    """
    DEMETERのCSVを「1035列（メタ11 + 周波数1024）」として壊れずに読み込む。
    元CSVの先頭行ヘッダは11列しかなくてもOK。

    ポイント：
    - header=None にして、ヘッダ行も“データ扱い”にしない
    - skiprows=1 で元のヘッダ行を捨てる
    - names=1035列名 を与えて列数を固定する
    """
    freq_cols = make_freq_cols()
    all_cols = META_COLS + freq_cols  # 合計1035列

    df = pd.read_csv(
        csv_path,
        header=None,          # CSV内ヘッダを使わない
        skiprows=1,           # 先頭の元ヘッダ行（11列）を読み飛ばす
        names=all_cols,       # こちらで1035列名を与える
        engine="c",           # 基本は高速なCエンジン
        on_bad_lines="skip",  # 壊れた行があればスキップ（止まらない）
    )
    return df


def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """
    指定列を数値に変換する（変換できないものはNaNになる）。
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def build_datetime_series(df: pd.DataFrame) -> pd.Series:
    """
    メタ列から datetime を作る（欠損があれば NaT になる）。
    milsecond はミリ秒として加算。
    """
    # year..second は整数っぽいが欠損があるので一旦 float/NaN を許す
    base = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
            minute=df["minute"],
            second=df["second"],
        ),
        errors="coerce",
    )

    # milsecond（ミリ秒）を timedelta にして足す（欠損はNaTのまま）
    ms = pd.to_timedelta(df["milsecond"], unit="ms", errors="coerce")
    dt = base + ms
    return dt


def interpolate_datetime_methodB(dt: pd.Series) -> pd.Series:
    """
    方法B：datetime を連続量として線形補間し、欠損を埋める。
    - datetime64[ns] を “nsの数値” に変換して補間
    - 補間後に datetime に戻す
    """
    # datetime → ns数値（NaTはNaNにしたいので一旦floatへ）
    dt_ns = dt.astype("int64").astype("float64")  # NaTは最小値になるので後でNaN化する
    # NaTをNaNに直す（NaTのint64は -9223372036854775808 ）
    dt_ns[dt.isna()] = np.nan

    # 線形補間（両端も埋める）
    dt_ns_filled = pd.Series(dt_ns).interpolate(method="linear", limit_direction="both")

    # まだNaNが残るなら（全欠損など）ここで落とす
    if dt_ns_filled.isna().any():
        raise ValueError("datetime interpolation failed: still contains NaN")

    # ns数値 → datetime に戻す
    dt_filled = pd.to_datetime(dt_ns_filled.astype("int64"), unit="ns")
    return dt_filled


def rebuild_discrete_from_datetime(df: pd.DataFrame, dt: pd.Series) -> None:
    """
    補完済みdatetimeから離散量（year..second,milsecond）を作り直して上書きする。
    """
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["second"] = dt.dt.second
    # ミリ秒は microsecond から作る（0〜999）
    df["milsecond"] = (dt.dt.microsecond // 1000).astype(int)


def interpolate_continuous_meta(df: pd.DataFrame) -> None:
    """
    連続量（lat/lon/mlat/mlon）を線形補間で埋める（上下の平均に相当）。
    """
    for c in ["lat", "lon", "mlat", "mlon"]:
        df[c] = df[c].interpolate(method="linear", limit_direction="both")


def compute_band_mean(df: pd.DataFrame, f_low: float, f_high: float) -> pd.Series:
    """
    1.7kHz帯域（指定範囲）の周波数ビン列を取り出し、行方向平均で1本に要約する。
    """
    cols = band_freq_cols(f_low, f_high)
    if len(cols) == 0:
        raise ValueError("band columns not found (check f_low/f_high or freq definition)")

    # 帯域の平均（各行で6列平均）
    return df[cols].mean(axis=1)


def append_step1_summary(summary_path: Path, row: dict) -> None:
    """
    1ファイル分のサマリをCSVに追記する（膨大ファイルでもメモリを食わない）。
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])

    # 初回はヘッダ付き、2回目以降はヘッダ無しで追記
    write_header = not summary_path.exists()
    df_row.to_csv(summary_path, mode="a", header=write_header, index=False, encoding="utf-8")


def step1_process_one_file(
    csv_path: Path,
    out_dir: Path,
    summary_path: Path,
    logger: logging.Logger,
    f_low: float = 1621.09375,
    f_high: float = 1718.75,
) -> Path:
    """
    Step1の「1ファイル処理」本体。
    - 読み込み（1035列）
    - 数値化
    - 欠損行フラグ
    - datetime生成→補間（方法B）→離散列再生成
    - 位置補間
    - 1.7kHz帯域平均を作成
    - 必要最小限だけ保存
    - サマリ追記

    戻り値：出力ファイルのパス
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 読み込み
    df = read_demeter_csv_as_1035cols(csv_path)

    # 2) 数値化（メタ11列 + 周波数1024列）
    freq_cols = make_freq_cols()
    to_numeric_inplace(df, META_COLS + freq_cols)

    df = df.copy()

    # 3) 欠損行フラグ（元々メタが全欠損の行を True にする）
    meta_all_nan = df[META_COLS].isna().all(axis=1)
    df["is_filled"] = meta_all_nan

    # 4) datetime作成（欠損はNaT）
    dt = build_datetime_series(df)

    # 5) 方法B：datetime を線形補間して欠損を埋める
    dt_filled = interpolate_datetime_methodB(dt)

    # 6) 補完済みdatetimeから離散列を作り直す（ここが方法Bの肝）
    rebuild_discrete_from_datetime(df, dt_filled)

    # 7) 位置（連続量）を補間
    interpolate_continuous_meta(df)

    # 8) 1.7kHz帯域平均（1621.09375〜1718.75Hz）を1本作る
    band_mean = compute_band_mean(df, f_low, f_high)
    df_out = pd.DataFrame(
        {
            "datetime": dt_filled,
            "lat": df["lat"],
            "lon": df["lon"],
            "mlat": df["mlat"],
            "mlon": df["mlon"],
            "E_1700band_mean": band_mean,
            "is_filled": df["is_filled"].astype(bool),
        }
    )

    # 9) 出力ファイル名（元名 + _step1.csv）
    out_path = out_dir / f"{csv_path.stem}_step1.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    # 10) サマリ追記（品質確認用）
    row = {
        "file": csv_path.name,
        "rows": int(len(df_out)),
        "meta_all_nan_rows": int(meta_all_nan.sum()),
        "meta_all_nan_ratio": float(meta_all_nan.mean()),
        "band_low_hz": f_low,
        "band_high_hz": f_high,
        "band_bins_count": int(len(band_freq_cols(f_low, f_high))),
        "out_file": str(out_path.name),
    }
    append_step1_summary(summary_path, row)

    logger.info(f"Step1 saved: {out_path.name} (rows={len(df_out)})")
    return out_path
