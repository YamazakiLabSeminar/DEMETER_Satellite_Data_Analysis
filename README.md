# プロジェクト
0. 基盤：paths/config/logging/膨大な数のCSVを回す骨格
1. 列名統一＋欠損補完＋1.7kHz帯域平均抽出＋数値型化＋軽量保存
- 形式を統一して（新ヘッダ）
- メタ欠損を“物理的に一貫する形で”補完し
- 1.7kHz帯域を平均して1本の指標にし
- 数値型として安定化した軽量データを作る

# 2.環境
Visual Studio Code Python 3.12.1
- 使用パッケージ：
	- pandas 2.2.1
 	- numpy 1.26.4
 	- matplotlib 3.8.0

# 3.フォルダ構成
ローカルでのプロジェクトのは以下のように：
```text
卒研解析/  （プロジェクトの親フォルダ）
├─ README.md                # 使い方メモ（自分用でOK）
├─ requirements.txt         # 必要ライブラリ一覧
├─ .gitignore               # 出力や巨大データをGitに入れない設定
│
│
├─ configs/
│  └─ config.yaml           # 実験条件（距離330km、4時間、窓幅±50s、閾値など）
│
└─ src/                     # Pythonコード本体
	├─ __init__.py
	├─ main.py               # 入口（ここ叩けば一連が動く）
	├─ paths.py              # パス管理（data/rawとかを一括で扱う）
	├─ io_demeter.py         # DEMETER CSV 読み込み
	├─ eq_catalog.py         # 地震カタログ処理・デクラスタリング
	├─ orbit_extract.py      # 地震軌道抽出（330km & 4h）
	├─ timeseries.py         # 切り出し・SEA・移動平均
	├─ anomaly.py            # 相関・異常判定
	└─ eval_molchan.py       # 警報率/予知率/Molchan

```

入力データは以下のように：
```text
inputs/	（入力データの親フォルダ）
├─ 00_EFdata_test/
├─ EFdata/
│ 
├─ external/				# Kp指数、地震カタログなど外部データ
│ 
└─ interim/					# 中間生成物（容量が許せば）
```

出力データは以下のように：
```text
outputs/		(出力データの親フォルダ)
├─ tables/					# CSV結果（地震リスト、抽出軌道リストなど）	
│ 
├─ figures/					# PNG図（Wordに貼る用）
│ 
├─ reports/					# まとめ（任意）
│
└─ logs/
   └─ run_YYYYMMDD_HHMM.log # 実行ログ（落ちたファイル等を記録）
```

- 解析を行う前に、まず、フォルダ構成の構築が大事である。必ず、これからやる。

# 0-1.パス管理
## 0-1-1 動作確認
簡単なコードで、Root(親フォルダ)の設定及び確認を```src/main.py```で行う。
```
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
print("Project root:", ROOT)
print("Now:", datetime.now())

```
- ```__file__```は「今実行しているこのPythonファイルの場所」を表す特別な変数。
- ```Path(__file__)```はファイル場所を```Path```に変換して、パスとして扱う。
- ```.resolve()```は「本当の絶対パス」に直す。
	- 例えば、相対パスやリンクが混じってても、ちゃんと完全なパスにしてくれる。
- ```.parents[1]```
	- ```parents```は「親フォルダをたどる」ためのもの。
	- ```parents[0]```は、1つ上のフォルダ
	- このフォルダ構成の場合、```parents[0]```→```src/```、```parents[1]```→```プロジェクトの根（卒業研究/）```

まとめて、このコード文の意味は：
```

「プロジェクトの一番上のフォルダをRootという名前で覚える」

```
という意味をする。

## 0-1-2 パスの設定
``` paths.py ```で、フォルダ構造のように、「どこにデータがあって、どこに出力するか」を全部一貫して扱える。
### 1. パス設定

<details><summary>サンプルコード</summary>

```
from pathlib import Path

# 1. プロジェクトのROOT（コードやconfigsを置く場所）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. 生データの場所（SSD1）←ここだけ自分の環境に合わせて書き換える
RAW_ROOT = Path(r"E:\DEMETER_RAW")  # 例：SSD1

# 3. 出力の場所（SSD2）←ここだけ自分の環境に合わせて書き換える
OUT_ROOT = Path(r"F:\DEMETER_OUT")  # 例：SSD2

# --- よく使うフォルダ（RAW側）---
RAW_DIR = RAW_ROOT / "csv"              # 例：E:\DEMETER_RAW\csv に全軌道CSVがある想定
EXTERNAL_DIR = RAW_ROOT / "external"    # 例：Kpや地震カタログをここに置く
INTERIM_DIR = OUT_ROOT / "interim"      # 中間生成物（容量が許せば）
TABLES_DIR = OUT_ROOT / "tables"        # 結果CSV
LOGS_DIR = OUT_ROOT / "logs"            # ログもSSD2にまとめる（推奨）
FIGURES_DIR = OUT_ROOT / "figures"      # Word貼り付け用PNGもここ（推奨）

def ensure_dirs() -> None:
    """必要フォルダを全部作る（既に存在してもOK）。"""
    for d in [EXTERNAL_DIR, INTERIM_DIR, TABLES_DIR, LOGS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

```

</details>

``` def ensure_dirs() -> None: ```
- 「フォルダがなければ作成する」ための関数を定義した。

``` 
for d in [ . . .]:
	d.mkdir(parents=True, exist_ok=True)
```
- ``` parents=True ```：途中のフォルダもまとめて作る。
- ``` exist_ok=True ```：すでにあってもエラーにしない。

### 2. 動作確認
``` src/main.py ```を更新てフォルダが作成できたかできなかったのテストをする。次に置き換える：
```
from paths import PROJECT_ROOT, RAW_DIR, OUT_ROOT, TABLES_DIR, FIGURES_DIR, LOGS_DIR, ensure_dirs

def main():
    ensure_dirs()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("OUT_ROOT:", OUT_ROOT)
    print("TABLES_DIR:", TABLES_DIR)
    print("FIGURES_DIR:", FIGURES_DIR)
    print("LOGS_DIR:", LOGS_DIR)

if __name__ == "__main__":
    main()

```

# 0-2. ログ入れ
巨大データ解析は、「どこで落ちたか、「何件処理したか」が命なので、ここを固める必要がある。
## 0-2-1 ```src/logger_setup.py```を新規作成

<details><summary>サンプルコード</summary>

```
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir: Path, name: str = "demeter") -> logging.Logger:
    """
    log_dir 配下にログファイルを作り、コンソール(画面)にも表示するロガーを返す。
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # 例: run_20260126_153012.log みたいなファイル名になる
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 二重に同じログが出ないようにする（再実行時に重要）
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. ログファイルへ出力する設定
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    # 2. 画面（ターミナル）へ出力する設定
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized")
    logger.info(f"log file: {log_path}")

    return logger

```

</details>

## 0-2-2 ログ対応確認
```
from paths import LOGS_DIR, ensure_dirs
from logger_setup import setup_logger


def main():
    ensure_dirs()
    logger = setup_logger(LOGS_DIR)

    logger.info("Start analysis (step0)")
    logger.info("This is a test log message.")
    logger.warning("This is a warning example (not an error).")
    logger.info("Finish step0")


if __name__ == "__main__":
    main()

```

# 0-3. 膨大な数のCSVを回す骨格
この段階では、
- CSVを昇順で列挙
- 1ファイルずつ処理（ダミー処理）
- 途中で落ちても止まらない
- 途中再開できる機能を追加する（前回どこまでやったかを記録する）
- 何件成功/失敗したがをログに残す。

## 0-3-1. ```src/io_utils.py```（ファイル列挙+チェックポイント）

<details><summary>サンプルコード</summary>

```
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import logging
import traceback


def iter_csv_files(folder: Path) -> list[Path]:
    """
    folder直下のCSVをファイル名昇順で返す。
    （膨大な数でも、まずは一覧を固定したいので list で返す）
    """
    return sorted(folder.glob("*.csv"))


def load_checkpoint(path: Path) -> set[str]:
    """
    既に処理済みのファイル名一覧を読み込む。
    （チェックポイントが無ければ空集合）
    """
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                done.add(name)
    return done


def append_checkpoint(path: Path, filename: str) -> None:
    """
    1件処理できたら、そのファイル名をチェックポイントに追記する。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(filename + "\n")


def safe_run_one(logger: logging.Logger, func, *args, **kwargs) -> bool:
    """
    関数を安全に実行。失敗しても止めず、例外をログに残す。
    """
    try:
        func(*args, **kwargs)
        return True
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        logger.error(traceback.format_exc())
        return False
```

</details>

## 0-3-2. ``` src/process_one_file.py```(現段階では軽いダミー処理)
```
from __future__ import annotations

from pathlib import Path


def process_one_csv(csv_path: Path) -> None:
    """
    1つのCSVに対する処理（今はダミー）。
    本解析ではここに読み込み・計算・出力を入れる。
    """
    # ダミー：ファイルが存在することだけ確認
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
```

## 0-3-3. ```src/main.py```(膨大な数受けの回し方)
- スキップ機能があるから、途中で止まっても次回すぐ続きから。
- ログを100件ごとにまとめるので、ファイル数が多くてもログが爆発しにくい。
- 各ファイルの処理は ``` process_one_csv() ```に閉じ込めるので、解析が進んでも``` main ```が汚れない。

<details><summary>サンプルコード</summary>

```
from __future__ import annotations

from paths import RAW_DIR, LOGS_DIR, TABLES_DIR, ensure_dirs
from logger_setup import setup_logger
from io_utils import (
    iter_csv_files,
    load_checkpoint,
    append_checkpoint,
    safe_run_one,
)
from process_one_file import process_one_csv


def main():
    ensure_dirs()
    logger = setup_logger(LOGS_DIR)

    logger.info(f"RAW_DIR = {RAW_DIR}")

    # 1 CSVを列挙（膨大でも、まず件数を把握）
    csv_files = iter_csv_files(RAW_DIR)
    logger.info(f"Found {len(csv_files)} csv files.")

    # 2 チェックポイント（処理済み）を読み込む
    checkpoint_path = TABLES_DIR / "checkpoint_done.txt"
    done = load_checkpoint(checkpoint_path)
    logger.info(f"Already done: {len(done)} files (from checkpoint).")

    ok = 0
    ng = 0
    skipped = 0

    # 3 1ファイルずつ処理
    total = len(csv_files)
    for i, csv_path in enumerate(csv_files, start=1):
        name = csv_path.name

        # 3-1 既に終わってるならスキップ（途中再開の要）
        if name in done:
            skipped += 1
            continue

        logger.info(f"[{i}/{total}] Processing: {name}")

        # 3-2 落ちても止まらない安全実行
        success = safe_run_one(logger, process_one_csv, csv_path)

        if success:
            ok += 1
            append_checkpoint(checkpoint_path, name)  # ここが再開のカギ
            done.add(name)  # 同じ実行中に重複処理しないため
        else:
            ng += 1

        # たまに進捗まとめ（ログが多すぎるのを防ぐ）
        if (ok + ng) % 100 == 0:
            logger.info(f"Progress: success={ok}, failed={ng}, skipped={skipped}")

    logger.info(f"Done. success={ok}, failed={ng}, skipped={skipped}")


if __name__ == "__main__":
    main()
```

</details>

# 1.新しいヘッダ付き+空白セル補完+1.7kHz帯域平均抽出+数値型化＋軽量保存
膨大なDEMETER軌道CSV（1ファイル＝1軌道）を、後段（正規化・SEA・相関・Molchan）でそのまま使えるように、
- 形式を統一して（新ヘッダ）
- 空白セルの補間
- 1.7kHz帯域サンプルを平均して1本の指標にし
- 数値型として安定化した軽量データを作る

## 1.1 読み込み（1035列として壊れない読み込み）
- 入力CSVは各行 1035列（メタ11 + 周波数1024）
- 先頭行ヘッダが11列でも崩れないように、読み込み時に names=1035本の新ヘッダを与えて読む（生データは書き換えない）

## 1.2 新しいヘッダを付与
- メタ列：```year, month, day, hour, minute, second, milsecond, lat, lon, mlat, mlon```
- 周波数列：19.53125Hz〜20000Hz を 19.53125Hz刻みで1024本

## 1.3 メタ11列の欠損補完（datetimeを補完して離散列を再生成）
### 1.3.1 欠損していない行だけを使って datetime を作る
- year〜second + milsecond を使って datetime を作成（欠損行は一旦 NaT）

### 1.3.2 datetime を連続量として補完（線形補間）
- datetime を数値（例：UNIX時間やns）に変換して
- 欠損している行の時刻を 線形補間（上下の中間を取る）
- 補完結果を datetime に戻す

### 1.3.3 補完した datetime から離散量を“作り直す”
- datetime からyear, month, day, hour, minute, second, milsecond を再計算して列に上書き
	- 平均で生じる「month=6.5」みたいな不正が起きない

### 1.3.4 位置（lat/lon/mlat/mlon）の補完
- lat, lon, mlat, mlon は連続量として線形補間（上下平均）
	- （任意）is_filled フラグを作って「元々欠損だった行」を記録

## 1.4 数値型へ統一
- lat, lon, mlat, mlon は float
- 周波数スペクトルは float（可能なら float32 にして軽量化）
- datetime は datetime型のまま保持（時系列解析が楽）

## 1.5 1.7kHz帯（1621.09375〜1718.75 Hz）を抽出し、平均して1本の指標に要約
- 対象はビン 83〜88（6本）
- その6列を行方向に平均して E_1700band_mean（仮名） を作る
	- （列は最終的にこの1本だけ残す方針）

## 1.6 軽量データとして保存（SSD2）
- 生データは書き換えない
- 1軌道ファイルから保存するのは 必要最小限：
	- datetime
	- lat, lon, mlat, mlon
	- E_1700band_mean
- 出力先例: ```outputs/interim/step1_extracted/<元ファイル名>_E1700band.csv

## 1.7 品質サマリ
膨大数でも管理できるよう、1ファイルごとに軽い統計を集める：
- 行数
- メタ欠損率（補完前）
- 補完後にNaNが残っていないか
- 抽出帯域（83〜88）を使ったことの記録

## 1.8 コード
### 1.8.1 ```src/step1_demeter.py```（新規）

<details><summary>サンプルコード</summary>

```
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
    dt_ns = dt.view("int64").astype("float64")  # NaTは最小値になるので後でNaN化する
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
```

</details>

### 1.8.2 ```src/process_one_file.py``` （更新）
```
from __future__ import annotations

from pathlib import Path
import logging

from paths import INTERIM_DIR, TABLES_DIR
from step1_demeter import step1_process_one_file


def process_one_csv(csv_path: Path, logger: logging.Logger) -> None:
    """
    Step1: 1ファイル処理の呼び出し口。
    """
    out_dir = INTERIM_DIR / "step1_extracted"              # 出力フォルダ
    summary_path = TABLES_DIR / "step1_summary.csv"        # サマリ（追記）

    # Step1処理（例外は呼び出し側の safe_run_one がログに落として止めずに進む）
    step1_process_one_file(
        csv_path=csv_path,
        out_dir=out_dir,
        summary_path=summary_path,
        logger=logger,
        f_low=1621.09375,
        f_high=1718.75,
    )
```

### 1.8.3 ```src/main.py```（Step1実行用に更新）

<details><summary>サンプルコード</summary>

```
from __future__ import annotations

from paths import RAW_DIR, LOGS_DIR, TABLES_DIR, ensure_dirs
from logger_setup import setup_logger
from io_utils import iter_csv_files, load_checkpoint, append_checkpoint, safe_run_one
from process_one_file import process_one_csv


def main():
    ensure_dirs()
    logger = setup_logger(LOGS_DIR)

    logger.info("=== Step1 start ===")
    logger.info(f"RAW_DIR = {RAW_DIR}")

    csv_files = iter_csv_files(RAW_DIR)
    logger.info(f"Found {len(csv_files)} csv files.")

    # Step1用のチェックポイント（Step0.5とファイル名を分ける）
    checkpoint_path = TABLES_DIR / "checkpoint_step1_done.txt"
    done = load_checkpoint(checkpoint_path)
    logger.info(f"Already done (checkpoint): {len(done)} files.")

    ok = 0
    ng = 0
    skipped = 0
    total = len(csv_files)

    for i, csv_path in enumerate(csv_files, start=1):
        name = csv_path.name

        if name in done:
            skipped += 1
            continue

        logger.info(f"[{i}/{total}] Step1 processing: {name}")

        # safe_run_one が例外を捕まえてログに残し、止めずに次へ進む
        success = safe_run_one(logger, process_one_csv, csv_path, logger)

        if success:
            ok += 1
            append_checkpoint(checkpoint_path, name)
            done.add(name)
        else:
            ng += 1

        # ログが多すぎないように100件ごとにまとめ表示
        if (ok + ng) % 100 == 0:
            logger.info(f"Progress: success={ok}, failed={ng}, skipped={skipped}")

    logger.info(f"=== Step1 done === success={ok}, failed={ng}, skipped={skipped}")


if __name__ == "__main__":
    main()
```

</details>

## 1.9 チェックポイント
- outputs/interim/step1_extracted/（あなたの INTERIM_DIR に依存）に
DMT_...._step1.csv が大量に生成される。
- outputs/tables/step1_summary.csv に1ファイル1行のサマリが追記される。
- outputs/tables/checkpoint_step1_done.txt に処理済みファイル名が溜まる（途中再開OK）
- ログに「どれで落ちたか（traceback）」が残る
- Step1出力CSVの列が 7列になっているか
```datetime, lat, lon, mlat, mlon, E_1700band_mean, is_filled```
- E_1700band_mean が NaNだらけになっていないか（帯域抽出が成功しているか）
- 2回目実行で skipped が増えて高速に終わるか（チェックポイントOK）

# 2. 正規化：ビン分け+CDF正規化
## 2.1 目的
Step 1で作った```E_1700band_mean```はそのままだと、
- 地磁気経緯度
- 季節
- 磁気嵐活動(Kp指数)

の影響を強く受けて、背景レベルが場所と条件で変わる。したがって、**条件ごとに背景分布を揃えて、値を「同じスケール（0~1）に変換する」**必要がある。言いかえると、**比較可能な相対強度にする。**

## 2.2 Step2の入力
- Step1出力(各軌道)
    - ```datetime, lat, lon, mlat, mlon, E_1700band_mean, is_filled```
- Kpデータ（2004-2010、絵時間刻み）
    - ```year,month,day,hour,minute,sec,milsec,kp```（naは使わない）
- 時刻基準：**UTC**
- 結合：datetimeに最も近いKp(nearest)

## 2.3 具体作業
1. Kpデータを読み込んで「datetime列」を作る。
    - 両側で作成する。

2. Kpデータにおける文字列(Kp指数列)を数値化する。
    Kp指数を**数値化**する。(kp_str → kp_num) 
    - 変換例：
        - ```"2"``` → 2.0
        - ```"2+"``` → 2 + 1/3 = 2.333...
        - ```"2-"``` → 2 - 1/3 = 1.666...
    - これで連続値kp_numが得られる。

3. 数値化したKp指数をカテゴリ
kp_num → kp_cat
    - 静か：```0, 0+, 1-, 1```
        - kp_num <= 1.0
    - 普通: ``` 1+, 2-, 2, 2+```
        - 1.0 < kp_num < 8/3 (8/3=2.666...は3-に相当)
    - 擾乱: ``` 3-以上 ```
        - kp_num >= 8/3

4. Step1データにKpをnearest結合する
    - Step1 のdatetimeと、Kpのdatetime_kpを比較し
    - 各行に「最も近い」Kpを付与する
    - 付与する列は最低限:
        - kp_str，kp_num，kp_cat

5. ビン分け(bin_idを作る)
    - 地磁気緯度: 2[deg]step
    - 地磁気経度: 5[deg]step
    - 季節：
        - 春(3-5)
        - 夏(6-8)
        - 秋(9-11)
        - 冬(12-2)
    - kp指数```kp_cat```: 上で決めたカテゴリ

    以上より、```bin_id```を作る.
    - ```bin_id = (mlat_bin, mlon_bin, season, kp_cat)```
        - このbin_idごとに背景分布を作る。

6. CDF正規化(E_norm)

    各```bin_id```内での```E_1700band_mean```の分布に基づき、E_norm∊[0, 1]を計算して列として付与する。

    膨大なデータのため、以下の手順で進める：
    1. binごとの```min/max/count```集計
    2. bin事の固定本数ヒストグラム作成(例: 256bins)
    - パス1の```min_E~min_E```を基づいて、固定本数(例256)の区間に分けて```hist[0..255]をbinごとに作る。
    3. ヒストグラムの累積で各点の```E_norm```を計算して保存する。
    - 各binのhistから累積(CDF)を計算して、各データ点の```E```が入る区間を求め、```E_norm(0~1)```をつけて、正規化済みデータとして保存
    - (オプション操作)以下の確認条件を満たしたら：
        - ```checkpoint_step2_done.txt```にStep1の全ファイル名が入っている。
        - ```outputs/interim/step2_normalized/``` に出力ファイル数が揃っている（= Step1入力数と一致、または失敗分がログに明記）
        - ```tables/step2_bin_counts.csv``` などが生成されている（統計が完成している）
    読み取った100個分のStep1出力を削除する。


7. Step2の出力(正規化済みデータ)

    各軌道ファイルごとにSSDへ保存：
    - ```datetime, lat, lon, mlat, mlon```
    - ```kp_str, kp_num, kp_cat```
    - ```mlat_bin, mlon_bin, season, bin_id```
    - ```E_1700band_mean```
    - ```E_norm(0~1)```
    - ```is_filled```

    保存先例: ```outputs/interim/step2_normalized/```

    加えて品質確認デーブル:
    - ```tables/step2_bin_counts.csv```（binの件数、min/max）
    - ```tables/checkpoint_step2_done.txt```（途中再開）


## 2.4 コード実装
1. Step2における条件設定

```configs/step2_normalization.yaml```(新規)

<details><summary>サンプルコード</summary>

``` python
# Step2: Normalization (binning + Kp merge + CDF normalization)

kp:
  csv_filename: "kpデータ_ALL(csv).csv"
  join_method: "nearest"          # nearest固定（あなたの方針）
  tolerance_hours: 2              # merge_asofの許容差（±2h）
  numeric_rule: "thirds"          # '+' '-' を 1/3 として数値化

  # Kpカテゴリ境界（あなたの定義を数値化で表現）
  # 静か: kp_num <= 1.0
  # 普通: 1.0 < kp_num < 8/3
  # 擾乱: kp_num >= 8/3
  quiet_max: 1.0
  normal_max: 2.6666666667        # 8/3 (=2.666...) これ未満が普通

binning:
  mlat_step_deg: 2
  mlon_step_deg: 5
  # mlonを0..360に丸めるか（負のmlon対策）
  mlon_to_0_360: true

season:
  spring_months: [3, 4, 5]
  summer_months: [6, 7, 8]
  autumn_months: [9, 10, 11]
  winter_months: [12, 1, 2]

cdf:
  hist_bins: 256                  # ヒスト分割数（近似CDFの細かさ）
  eps_range: 1.0e-12              # min==max対策

io:
  step1_dirname: "step1_extracted"
  step2_dirname: "step2_normalized"
  checkpoint_filename: "checkpoint_step2_done.txt"
  bin_stats_filename: "step2_bin_counts.csv"
```

</details>