from pathlib import Path

# 1. プロジェクトのROOT（コードやconfigsを置く場所）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. 生データの場所（SSD1）←ここだけ自分の環境に合わせて書き換える
RAW_ROOT = Path(r"F:")  # 例：SSD1

# 3. 出力の場所（SSD2）←ここだけ自分の環境に合わせて書き換える
OUT_ROOT = Path(r"E:")  # 例：SSD2

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
