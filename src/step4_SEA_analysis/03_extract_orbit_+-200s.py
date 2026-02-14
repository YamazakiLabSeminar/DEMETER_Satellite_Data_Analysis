# -*- coding: utf-8 -*-
# 03_extract_orbit_±200s.py
###################################################################################################
###################################################################################################
# Python script to 最接近事時刻から±200sデータを有する軌道を確認し、抽出する。
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
# 1. closest_time_dest_ver*.csvをDataFrame(df)として輸入する。
# 2. dfから"orbit_num", "min_index", "closest_time", を読み取る。
# 3. "orbit_num"で、orbit_data_for_sea_analysis_candidate_distフォルダに保存されている同名ファイルを開き。
# 4. 各軌道データ内で、"min_index"から前後97個サンプルがあるかないかを確認する。
# 5. ある場合、読み取っている軌道データファイルをorbit_data_for_sea_analysisフォルダにコピーさせる。ない場合、スキップ。
#
#**************************************************************************************************
# Structure of this script:
#==================================================================================================
# [1.   Importing the modules]
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
#--------------------------------------------------------------------------------------------------
# [2.   Setting Path]
# 2-1.  Setting Path and Directory
MATCH_TABLE_PATH = Path(r"E:\tables\orbit_eq_match_with_closest_time_dist\closest_time_dist_ver2.csv")
SEA_CAN_DIST_DIC = Path(r"E:\interim\orbit_data_for_sea_analysis_candidate_dist")
OUTPUT_PATH = Path(r"E:\tables\orbit_with_+-200s")
NEW_ADDRESS = Path(r"E:\interim\orbit_data_for_sea_analysis_+-200s")
# 2-2.  パス存在した場合の対策
NEW_ADDRESS.mkdir(parents=True, exist_ok=True)
#--------------------------------------------------------------------------------------------------
# [3.   Setting Parameter]
###################################################################################################
before = 100     # 最接近時刻から前100個サンプル(-100row)
after = 100      # 最接近時刻から後100個サンプル(+100row)

USECOLS = ["eq_id","orbit_num", "min_index"]    # Match Tableで使用される列
have_count = 0      # 最接近時刻から+-200sにデータがある軌道データファイル数
###################################################################################################
# [4.   Preparation]
# 4-1.  Importing the match table with columns "orbit_nun", "min_index"as a data frame.
df = pd.read_csv(MATCH_TABLE_PATH, usecols=USECOLS)
# 4-2.  Converting columns"orbit_nun" into string
df[USECOLS[1]] = df[USECOLS[1]].astype("string")
# 4-3.  Information of Dataframe.
print("[Info]   DataFrame of MATCH_TABLE:\n==================================================")
df.info()
print("--------------------------------------------------")
print(df)
print("**************************************************\n")
# 
list1 = []
#--------------------------------------------------------------------------------------------------
# [5.   最接近時刻から+-200sのサンプルを有している軌道データを引っ越す。]
# 5-1.  行順番にmatch tableの"orbit_num","eq_id","min_index"を読むループを作成。
for i in range(len(df)):
    eq_id = df[USECOLS[0]].iloc[i]
    orbit_num = df[USECOLS[1]].iloc[i]
    min_index = df[USECOLS[2]].iloc[i]
    # 5-2.  候補フォルダにあるファイルを読めるためのファイル名を作成
    file_name = f"{orbit_num}_eq{eq_id}.csv"
    # 5-3.  候補フォルダ内に同名ファイルを開く
    df_ob = pd.read_csv(SEA_CAN_DIST_DIC / file_name)
    # 5-4.  DataFrameの情報を一回だけ示す。
    if i == 0:
        print("[Info]   DataFrame of READING ORBIT DATA FILE:\n==================================================")
        df_ob.info()
        print("--------------------------------------------------")
        print(df_ob)
        print("**************************************************")
    # 5-5.  "min_index"より、距離の最小値がある位置を探す。
    min_dist = df_ob["dist"].iloc[min_index]
    # 5-6.  Confirm the correction of min_index
    if i<10:
        print(min_dist)
    # 5-7.  min_indexから前後100個サンプルがあるかないかを調べる。
    n = len(df_ob)      # df_obの行数
    has_window = (min_index - before >= 0) and (min_index + after < n)

    # if has_window:
    #     window = df_ob.iloc[min_index - before : min_index + after + 1]
    # else:
    #     pass

    # 5-8.  あったら、新しいフォルダにそのファイルをコピーし保存する。
    if has_window:
        window = df_ob.iloc[min_index - before : min_index + after + 1].copy()
        window.to_csv(NEW_ADDRESS / file_name, index=False)
        row = df.iloc[i].to_dict()
        list1.append(row)
        have_count += 1
    else:
        pass

df_output = pd.DataFrame(list1, columns=USECOLS)
df_output.to_csv(OUTPUT_PATH / "SEA_analysis_eq_ob_ver3.csv", index=False)
#
#--------------------------------------------------------------------------------------------------
# [6.    処理結果を示す。]
# 総候補数：
print("[Info]   総候補数:", len(df))
# 最接近時刻から+-200sのサンプルを有している軌道数
print("[Info]   使用可能数", have_count)
#
#**************************************************************************************************
