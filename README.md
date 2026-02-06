# 1.プロジェクト
モデルによる異常の判断基準の一致性を保証するため，同様な方法や条件でデータを解析する必要がある． DEMETER衛星の運用期間（2004年~2010年）に発生したマグニチュードM4.8以上震源深さ40 kmの地震の，震央から距離約330 km以内を通過したDEMETER衛星軌道の1.7 kHz電場強度データを時系列解析する．電離圏の電場強度は，磁気緯度，磁気経度，季節，磁気嵐活動によって影響を受けるため，全衛星データに対して各条件にビン分け，スケーリングを同一させるためビン内に累積分布関数（CDF）で正規化する．SEA解析 より震央付近での電場強度変動を時系列変動として得られ，±50秒の移動平均で平滑なトレンドを得られる．さらに，ガウス関数フィッティングでモデル化し，四象限法を基づいて各地震軌道の変動とモデルの変動との相関係数を算出し，予知率と警報率を算出する．最後に，異常の閾値を変化しながらダイアグラムを作成する．

1. 基盤：paths/config/logging/膨大な数のCSVを回す骨格
2. 列名統一＋欠損補完＋1.7kHz帯域平均抽出＋数値型化＋軽量保存

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
│  └─ step2_normalization.yaml              # 正規化の設定条件（bin条件、kpデータ処理設定、cdf設定）
│
└─ src/                     # Pythonコード本体
	├─ __init__.py
	├─ logger_setup.py       # loggerの設定
	├─ paths.py              # パス管理（data/rawとかを一括で扱う）
	├─ io_utils.py         # DEMETER CSV 読み込み
	├─ eq_catalog.py         # 地震カタログ処理・デクラスタリング
	├─ config_loader_setup.py
	├─ process_one_file.py
	├─ read_demeter.py
	│
    ├─ step1/
    │   ├─ __init__.py   
    │   ├─ main.py
    │   └─ step1_demeter.py
    │
    ├─ step2/
    │   ├─ __init__.py
    │   ├─ main.py
    │   └─ step2_normalization.py
    └─ 

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
    - 電場強度データも同様に作成する。

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

5. ビン分け、bin_idを作る(ファイルを読むたび実行)
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
### 1. Step2における条件設定

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

---

### 2. configloaedの関数定義

``` src/config_loader.py```（新規）

<details><summary>実装コード</summary>

``` python
from __future__ import annotations

from pathlib import Path

def load_yaml_config(path: Path) -> dict:
    """
    YAML設定ファイルを読み込んで dict を返す。
    もし pyyaml が入っていなければ、分かりやすいエラーを出す。
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml が必要です。`pip install pyyaml` を実行してください。"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return cfg

```

</details>

- ```from __future__ import annotations```
    - Pythonの将来機能を先取りする宣言。
    - これを入れると、型ヒント（Path や dict など）の扱いが少し柔軟になって、型ヒント周りのトラブルが減りやすい。
    - 「書かなくても動くことが多いけど、書いておくと便利」系。

---

### 3. 正規化のコード（本体）

``` step2_normalization.py```(新規)

<details><summary>実装コード</summary>

``` python
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
        return "不明"
    if kp_num <= quiet_max:
        return "静か"
    if kp_num < normal_max:
        return "普通"
    return "擾乱"


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
        return "春"
    if month in summer:
        return "夏"
    if month in autumn:
        return "秋"
    if month in winter:
        return "冬"
    return "不明"


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

```

</details>

---

### 4. main文

``` step2_main.py ```(新規)

<details><summary>実装コード</summary>

``` python
from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """
    コマンドライン引数（オプション）を定義する。
    例：
      python -m src.step2.main
      python -m src.step2.main --config configs/step2_normalization.yaml
    """
    p = argparse.ArgumentParser(description="Step2: Kp merge + binning + CDF normalization")
    p.add_argument(
        "--config",
        type=str,
        default="configs/step2_normalization.yaml",
        help="Path to Step2 YAML config (relative to project root allowed).",
    )
    return p


def main() -> None:
    # ===== 0) 引数を読む =====
    args = build_argparser().parse_args()

    # ===== 1) プロジェクトルート（卒研解析/）を決める =====
    # このファイルは src/step2/main.py にある。
    # main.py → step2 → src → (プロジェクトルート) なので parents[2]。
    project_root = Path(__file__).resolve().parents[2]

    # ===== 2) あなたのフォルダ構成に合わせて主要パスを組む =====
    configs_dir = project_root / "configs"
    inputs_dir = project_root / "inputs"
    outputs_dir = project_root / "outputs"

    external_dir = inputs_dir / "external"
    interim_dir = inputs_dir / "interim"

    logs_dir = outputs_dir / "logs"
    tables_dir = outputs_dir / "tables"

    # ===== 3) configファイルのパスを確定 =====
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    # ===== 4) YAMLを読む（上流：config_loader） =====
    # ※あなたが持っている src/config_loader_setup.py を使う
    from ..config_loader_setup import load_yaml_config
    cfg = load_yaml_config(cfg_path)

    # ===== 5) loggerを作る（上流：logger_setup） =====
    from ..logger_setup import setup_logger
    logger = setup_logger(logs_dir, name="step2")

    # ===== 6) Step2入出力パスを config から決める =====
    # Step1出力は inputs/interim/ にある前提
    step1_dirname = cfg.get("io", {}).get("step1_dirname", "step1_extracted")
    step2_dirname = cfg.get("io", {}).get("step2_dirname", "step2_normalized")

    step1_dir = interim_dir / step1_dirname
    step2_out_dir = interim_dir / step2_dirname

    kp_csv_filename = cfg.get("kp", {}).get("csv_filename", "kpデータ_ALL(csv).csv")
    kp_csv_path = external_dir / kp_csv_filename

    # ===== 7) 実行情報をログに残す =====
    logger.info("=== Step2 start ===")
    logger.info(f"project_root = {project_root}")
    logger.info(f"config_path  = {cfg_path}")
    logger.info(f"step1_dir    = {step1_dir}")
    logger.info(f"kp_csv_path  = {kp_csv_path}")
    logger.info(f"step2_outdir = {step2_out_dir}")
    logger.info(f"tables_dir   = {tables_dir}")
    logger.info(f"logs_dir     = {logs_dir}")

    # ===== 8) Step2本体を呼ぶ =====
    from .step2_normalization import run_step2

    run_step2(
        step1_dir=step1_dir,
        kp_csv_path=kp_csv_path,
        out_dir=step2_out_dir,
        tables_dir=tables_dir,
        cfg=cfg,
        logger=logger,
    )

    logger.info("=== Step2 end ===")


if __name__ == "__main__":
    main()

```

</details>

---

# 3. SEA解析
## 3.1 目的
Step2で正規化済みデータ(E_norm)から、震央付近にある地震先行時系列変動を作成する。各地震イベントに対して、「震央最接近時刻(t=0)」基準で前後の時系列を切り出す。

## 3.2 アプローチ
### 1.解析条件
- 時刻はUTCに基準

- 時間分解能：DEMETER Survey modeで2.048秒

- 1 地震イベントに対して、地震発生4時間以内に震央から距離330km以内を通過する軌道を抽出する。ただし、330km区間における衛星の地磁気緯度の絶対値が>65°を含む地震イベントは除外する。
---

### 2. 地震カタログの前処理
- 2004年から2010年までの震源深さ40km以下、震度4.8以上の地震カタログから、地震発生日時、地震発生緯度/経度、震源深さ、地震震度を抽出する。
    - ["time","latitude","longitude","depth","mag",]

- 抽出したデータについて、ヘッダ名を変更し、インデックス列"eq_id"列を追加する。
    - 新しヘッダ["eq_id","datetime","lat","lon","depth","mag"]

- 新しいcsvファイルとして出力する。

- 新しく出力されたcsvファイルを読み込み、地震発生から4時間前の時刻を計算して、新しい列として追加する。
    - ヘッダ["eq_id","4hour_before","datetime","lat","lon","depth","mag"]

### 3. 地震軌道の抽出
#### 内容：
1. Step2出力(正規化後データ)の全csvファイルについて、
    - 軌道ファイルの最初と最後の日時```["datetime"]```を調べて、["ファイル名", "開始時刻", "終了時刻"]が乗せた軌道indexリストを作成し、```step3_orbit_index.csv```として出力する。

2. デクラスタリング後の本震のみのカタログから```["latitude", "longitude", "mag", "time"]```を地震発生緯度、経度、震度、地震発生日時として読み取る。

3. 各地震イベントについて、
    - step3_orbit_index.csvを読み取り、軌道ファイルの["開始時刻", "終了時刻"]時間区間が, 地震発生4時間前の時間帯内にあれば、その軌道の[ファイル名、軌道開始/終了時刻]と対応する[地震発生緯度/経度、震度、地震発生日時]、を抽出し、新規の候補リストに追加する。
        - [軌道ファイルの"終了時刻"]>[地震発生時刻]、かつ、[軌道ファイルの"開始時刻"]<[地震発生時刻-4h]

    - 候補リストにある軌道ファイル(step2出力)を読み、各サンプルの震央からの距離をヒュベニ公式(WGS84)で計算し、震央からの距離が<=330kmの区間を探す。

    - この区間があれば、そのまま候補リストに残し、この軌道の距離<=330 kmを満たす最初/最後のサンプル時刻、および衛星が距離<=330km区間内での震央からの最小距離を候補リストに追加する。
        - その区間がなければ、その軌道は不採用、次の候補へ。
        - ある1地震イベントに、全候補軌道が不採用ならば、その地震は「紐づけなし」にする。
        - 1地震イベントについて、候補した軌道が複数あれば、震央からの最近距離が最小の軌道1本を採用。

4. 候補リストを以下のようにcsvファイルとして出力する。
    - 地震イベントid、地震発生時刻、地震発生緯度、地震発生経度、軌道ファイル名、軌道開始時間、軌道終了時間、区間入り時間、区間出る時間、衛星が距離<=330km区間内での震央からの最小距離、を乗せる。
    - ["eq_time", "eq_lat", "eq_lon",
     "orbit_file", "orbit_start_time", "orbit_end_time", 
     "pass_time_start", "pass_time_end", 
     "closest_dis_km"]
---
#### 入力:
1. 地震カタログ：
- パス: ```"F:\external\eq_catalog\all_eq_declustring_30day_30km.csv"```

- 使用列:```["latitude", "longitude", "mag", "time"]```

- ["time"]は```datetime```型である。時間はUTCに基準したもの。
---

2. 正規化後データ(Step2):
- パス: ```"E:\interim\step2_normalized"```

- 使用列: ["datetime","lat","lon","mlat","mlon"]
---

#### 出力：
1. 軌道index表:
    - パース:```"E:\tables\step3_orbit_index.csv"```

    - ヘッダ: ["orbit_file","orbit_start_time","orbit_end_time"]

2. 地震-軌道紐づけ表：
    - パース：```"E:\tables\step3_orbit_map.csv"```

    - ヘッダ:
    - ["eq_time","eq_lat","eq_lon",
 "orbit_file","orbit_start_time","orbit_end_time",
 "pass_time_start","pass_time_end",
 "closest_dis_km"]

---
#### 確認ポイント：
- 紐づけなかった地震イベント数
- 複数軌道が該当した地震イベント数と、その場合に最小距離が最小の軌道を採用できているか確認。
---
### 3. 震央最接近時刻の計算
#### 内容
- 地震-軌道紐づけ表、正規化後データ、地震カタログを読み取る。

- 各地震イベントについて、衛星位置(lat/lon)と震央(lat/lon)の距離を、楕円体WGS84を使用した**ヒュベニ公式**で計算する。

- 最小距離となる時刻を、```t0 ```にする。
    - 最小距離 closest_dis_km は t0=closest_time における距離とする。

- ```t0```を地震-紐づけ表(```step3_orbit_map.csv```)に```["closest_time"]```として追加する。
---
#### 入力
- 正規化後データ(Step2):
```"E:\interim\step2_normalized"```

- 地震-軌道紐づけ表:
    ```"E:\tables\step3_orbit_map.csv"```

- 地震カタログ：
```"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_200407-201012.csv"```
---
#### 出力
- ```step3_orbit_map.csv```:
    - ヘッダ:
    ```["eq_id", "eq_time", "eq_lat", "eq_lon", "orbit_file", "pass_time_start", "pass_time_end", "orbit_datetime_start", "orbit_datetime_end", "closest_dis_km", "closest_time"]```

---
#### 確認ポイント
- ```"closest_dis_km"```が330km以内になっているか

- 接近点がファイル端に偏ってないか
---

