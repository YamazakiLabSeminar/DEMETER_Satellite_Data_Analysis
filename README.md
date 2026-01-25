# 1.DEMETER衛星のSurveyモードの電場データの解析
- 電場Surveyデータ(1132)を用いて解析を行う。[参考GitGub(曽根凪紗)](https://github.com/ElecScape/DEMETER-EFSurvey.git)
- DMETER衛星の情報およびデータについては、簡略にこの[ファイル](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/50e2140eb740177b159f508862e533630209cda7/DEMETE%E8%A1%9B%E6%98%9F%E8%A6%B3%E6%B8%AC%E3%83%87%E3%83%BC%E3%82%BF.md)に書いてある。

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

# 4.パス管理
## 4.1 動作確認
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

## 4.2 パスの設定
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

# 5. ログ入れ
巨大データ解析は、「どこで落ちたか、「何件処理したか」が命なので、ここを固める必要がある。
## 5.1 ```src/logger_setup.py```を新規作成
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

```
import logging
```
- Python標準の「ログ出力」道具。``` print() ```より強い（ファイルに残せる、重要度レベルがある）。

```
def setup_logger(log_dir: Path, name: str = "demeter") -> logging.Logger:
```
- 「ログを準備して、使える logger を返す関数」を作ってる。
- ```log_dir``` はログを保存するフォルダのパス。