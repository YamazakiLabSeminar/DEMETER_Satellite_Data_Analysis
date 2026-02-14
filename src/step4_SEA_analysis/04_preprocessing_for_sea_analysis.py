# -*- coding: utf-8 -*-
# extract_closest_time_dist.py
###################################################################################################
###################################################################################################
# Python script to SEA分析の前処理として、最接近事時刻から±200sのE_normを抽出し、テーブル化する。
# =================================================================================================
# Stucture of this script:
#==================================================================================================
# [1.       Importing the modules]
import pandas as pd
from pathlib import Path
#--------------------------------------------------------------------------------------------------
#
# [2.       Path Settings ]
#==================================================================================================
DATA_DIR = Path(r"E:\interim\orbit_data_for_sea_analysis_+-200s")
OUTPUT_DIR = Path(r"E:\tables\preprocessing_for_sea_analysis")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"File not found: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#------------------------------------------------------------------------------------------------
#
# [3.       SEA Analysis Settings ]
#==================================================================================================
TIME_WINDOW = 200  # seconds
TIME_RESOLUTION = 2  # seconds

# 出力用dataframe
time_columns = list(range(-TIME_WINDOW, TIME_WINDOW+TIME_RESOLUTION, TIME_RESOLUTION))
columns = ["eq_id", "orbit_num"] + [f"t_{t}" for t in time_columns]

sea_df = pd.DataFrame(columns=columns)
sea_df.info()
print(sea_df)
#------------------------------------------------------------------------------------------------
#
# [4.       SEA Analysis ]
#==============================================================================================
# //1. データディレクトリ内の全CSVファイルを取得//
file_names = sorted(DATA_DIR.glob("*.csv"))
# pprint.pprint(file_names)

# //2. ファイル名を分離する//
for file_name in file_names:
    stem = file_name.stem  # 例: DMT_N1_1132_34280.1_eq7018.csv -> DMT_N1_1132_34280.1_eq7018
    if "_eq" not in stem:
        print(f"[Skip] unexpected file name format: {file_name.name}")
        continue

    orbit_num, eq_part = stem.rsplit("_eq", 1)
    '''
    例: DMT_N1_1132_34280.1_eq7018 -> 
        orbit_num: DMT_N1_1132_34280.1, 
        eq_part: 7018
    '''
    if not eq_part.isdigit():
        print(f"[Skip] eq_id is not numeric: {file_name.name}")
        continue

    eq_id = int(eq_part)

# //3. 衛星データを読み込む//
    df_ob = pd.read_csv(file_name)
    E_norm = df_ob["E_norm"].tolist()
    expected_len = len(time_columns)
    if len(E_norm) < expected_len:
        E_norm = E_norm + [pd.NA] * (expected_len - len(E_norm))
    elif len(E_norm) > expected_len:
        E_norm = E_norm[:expected_len]

    row = {"eq_id": eq_id, "orbit_num": orbit_num}
    row.update({f"t_{t}": v for t, v in zip(time_columns, E_norm)})
    sea_df.loc[len(sea_df)] = row


print("[Info] SEA Analysis Result DataFrame:\n==================================================")
sea_df.info()
print("--------------------------------------------------")
print(sea_df)
print("**************************************************")

sea_df.to_csv(OUTPUT_DIR / "preprocessing_for_sea_analysis_results.csv", index=False)
