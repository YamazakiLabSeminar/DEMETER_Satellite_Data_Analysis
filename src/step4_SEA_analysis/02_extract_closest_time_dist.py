# -*- coding: utf-8 -*-
# extract_closest_time_dist.py
###################################################################################################
###################################################################################################
# Python script to 最接近事時刻と距離を抽出する。
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
#
#**************************************************************************************************
# Structure of this script:
#--------------------------------------------------------------------------------------------------
# [1.    Importing the modules]
import pandas as pd
from pathlib import Path
import math
from tqdm.auto import tqdm
import pprint
#
#
# [2.    Setting the directories/pathes]
# 2-1.  Setting the drectories/pathes
MAT_PATH = Path(r"E:\tables\orbit_eq_match\orbit_quake_distance_ver19.csv")
CAND_DIC = Path(r"E:\interim\orbit_data_for_sea_analysis_candidate_dist")
OUTPUT_DIC = Path(r"E:\tables\orbit_eq_match_with_closest_time_dist")
#
# 2-2.  パス存在しない場合の対策
OUTPUT_DIC.mkdir(parents=True, exist_ok=True)
#
#**************************************************************************************************
# [3.   Importing matching table]
# 3-1.  Importing matching table.
cols = ["eq_id","4h_before","occur_time","latitude","longitude","longitude_360","mag","orbit_file"]
df = pd.read_csv(MAT_PATH)
df.info()
#
#**************************************************************************************************
# [4.   Preparation for extracting]
# 4-1.  Creating a new Data Frame.
df_output = pd.DataFrame(columns=cols)
#
# 4-2.  Extracting "eq_id","4h_before","occur_time","latitude","longitude","depth","mag","orbit_file"
df_output[cols] = df[cols]
#
# 4-3.  軌道ファイル名を文字列に
df_output["orbit_file"] = df_output["orbit_file"].astype("string")
df_output.info()
#
# 4-4.  抽出されたデータ保存用のリストを作成する。
list1 = [[] for k in range(len(df_output))]
#
# 4-5.  処理されたファイル数のカウント
pro_count = 0
#
#**************************************************************************************************
# [5.   Extracting closest time distance of the individual candidate orbit data]
# Importing matching file name
for i in tqdm(range(len(df_output)), desc="Extract closest time dist", unit="file"):
    # ループ内に各軌道データを保存するリストを作成
    list2 = []
    # 軌道データのファイル名を抽出する。
    file_name = df_output["orbit_file"].iloc[i]
    # change the format of file name
    eq_id = df_output["eq_id"].iloc[i]
    stem = Path(file_name).stem
    base = stem.split("_step")[0]
    file_name = f"{base}_eq{eq_id}.csv"
    # 距離付きSEA解析用候補軌道データが保存されているフォルダ中、同じ名のデータを開き、輸入する。
    df_ob = pd.read_csv(CAND_DIC / file_name)
    # インデックスをリセット
    df_ob.reset_index(drop=True)
    # 最小値の行番号を取得する。
    min_index = int(df_ob["dist"].idxmin())
    #　最小距離の行を抽出する。
    row = df_ob.loc[min_index].to_dict()
    row["orbit_num"] = base
    row["min_index"] = min_index
    # list1[i]に挿入
    list1[i] = row
    # ループが一周回したら、カウントとが累積される。
    pro_count += 1
# マッチングされた総数
print("[Inof]   Total matched counts:",len(df))
# 処理されたデータ数
print("[Info]   Processed data counts:", pro_count)
# list1をdfに変換する。
data = pd.DataFrame(list1, columns=["bin_id","datetime","lat","lon","mlat","mlon","E_norm","dist","min_index","orbit_num"])
# 出力用dfとlistから変換されたdfを横方向に結合する。
outputone = pd.concat([df_output, data], axis=1)
# "orbit_file"列を削除する
outputone = outputone.drop(columns=["orbit_file"])
# dfの列の順番を変える。
outputone = outputone.reindex(columns=["eq_id","4h_before","occur_time","latitude","longitude","longitude_360","mag","orbit_num",
                                       "min_index","datetime","dist","bin_id","lat","lon","mlat","mlon","E_norm"])
# 列名を変える。
outputone = outputone.rename(columns={
    "datetime": "closest_time",
    "dist": "closest_dist"
})
# csvファイルとして出力する。
outputone.to_csv(OUTPUT_DIC / "closest_time_dist_ver4.csv", index=False)