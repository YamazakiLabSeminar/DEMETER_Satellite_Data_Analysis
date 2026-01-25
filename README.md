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

### 0-3-3. ```src/main.py```(膨大な数受けの回し方)
- スキップ機能があるから、途中で止まっても次回すぐ続きから。
- ログを100件ごとにまとめるので、ファイル数が多くてもログが爆発しにくい。
- 各ファイルの処理は ``` process_one_csv() ```に閉じ込めるので、解析が進んでも``` main ```が汚れない。

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

# 1.4 数値型へ統一
- lat, lon, mlat, mlon は float
- 周波数スペクトルは float（可能なら float32 にして軽量化）
- datetime は datetime型のまま保持（時系列解析が楽）

# 1.5 1.7kHz帯（1621.09375〜1718.75 Hz）を抽出し、平均して1本の指標に要約
- 対象はビン 83〜88（6本）
- その6列を行方向に平均して E_1700band_mean（仮名） を作る
	- （列は最終的にこの1本だけ残す方針）

# 1.6 軽量データとして保存（SSD2）
- 生データは書き換えない
- 1軌道ファイルから保存するのは 必要最小限：
	- datetime
	- lat, lon, mlat, mlon
	- E_1700band_mean
- 出力先例: ```outputs/interim/step1_extracted/<元ファイル名>_E1700band.csv

# 1.7 品質サマリ
膨大数でも管理できるよう、1ファイルごとに軽い統計を集める：
- 行数
- メタ欠損率（補完前）
- 補完後にNaNが残っていないか
- 抽出帯域（83〜88）を使ったことの記録


