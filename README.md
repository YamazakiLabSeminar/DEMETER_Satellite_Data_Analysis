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
├─ src/                     # Pythonコード本体
│  ├─ __init__.py
│  ├─ main.py               # 入口（ここ叩けば一連が動く）
│  ├─ paths.py              # パス管理（data/rawとかを一括で扱う）
│  ├─ io_demeter.py         # DEMETER CSV 読み込み
│  ├─ eq_catalog.py         # 地震カタログ処理・デクラスタリング
│  ├─ orbit_extract.py      # 地震軌道抽出（330km & 4h）
│  ├─ timeseries.py         # 切り出し・SEA・移動平均
│  ├─ anomaly.py            # 相関・異常判定
│  └─ eval_molchan.py       # 警報率/予知率/Molchan
│
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