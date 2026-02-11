# -*- coding: utf-8 -*-
# extract_required_data.py
###################################################################################################
###################################################################################################
# Python script to extract the required data from the orbit data which is included in the table of
# earthquake-orbits matching of the individual orbit.
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
from tqdm.auto import tqdm
import pprint
from collections import Counter
#
#
# [2.    Setting the directories/pathes]
MAT_PATH = Path(r"E:\tables\orbit_eq_match\orbit_quake_distance_ver15.csv")
ORBIT_DIC = Path(r"E:\interim\step2_normalized")
OUTPUT_DIC = Path(r"E:\interim\orbit_data_for_sea_analysis_candidate")
#
#
# [3.   Importing the table of earthquake-orbits matching]
# 3-1.  Importing the table of eq-orbits maching as a data frame.
df = pd.read_csv(MAT_PATH)
#
# ex.   Confirming the information of the data frame.
#print("[Info]   Table of eq-orbits matching\n==================================================")
#df.info()
#print("===================================================")
#print(df)
#
# 3-2.  Converting format of columns "orbit_file" from object into string
df["orbit_file"] = df["orbit_file"].astype("string")
#
# ex.   Confirming the information of the data frame.
#print("[Info]   Table of eq-orbits matching\n==================================================")
#df.info()
#print("===================================================")
#print(df)
#
#
# [4.   Extracting the required data which using for the SEA analysis]
# 出力用DataFrameの作成
df_output = pd.DataFrame(columns=["bin_id","datetime","lat","lon","mlat","mlon","E_norm"])

# # Creating the file name
missing_file = []
for i in tqdm(range(len(df)), desc="Extracting required data from matched orbit data", unit="file"):
    file_name = df["orbit_file"].iloc[i]

    # Creating the file path to orbit directory    
    file_path = ORBIT_DIC / file_name
    if not file_path.exists():
        missing_file.append(str(file_name))
        continue
        #raise FileNotFoundError(f"Orbit file not found: {file_path}")
    
    # Importing the orbit data 
    df_orbit = pd.read_csv(file_path)

    # ex.   Confirming the information of the data frame.
    #print("[Info]   Table of imported sat-data\n==================================================")
    #df_orbit.info()
    #print("===================================================")
    #print(df_orbit)

    # Extracting the required data from orbit df
    df_output["bin_id"] = df_orbit["bin_id"]
    df_output["datetime"] = df_orbit["datetime"]
    df_orbit["lat"] = df_orbit["lat"].astype(float).round(8)
    df_output["lat"] = df_orbit["lat"]
    df_orbit["lon"] = df_orbit["lon"].astype(float).round(7)
    df_output["lon"] = df_orbit["lon"]
    df_output["mlat"] = df_orbit["mlat"]
    df_output["mlon"] = df_orbit["mlon"]
    df_output["E_norm"] = df_orbit["E_norm"]
    
    # ex.   Confirming the information of the data frame.
    #print("[Info]   Table of output\n==================================================")
    #df_output.info()
    #print("===================================================")
    #print(df_output)

    #　Creating expoiting file name.
    eq_id = df["eq_id"].iloc[i]
    stem = Path(file_name).stem     # 拡張子除去
    base = stem.split("_step")[0]   # .1まで残る
    out_name = f"{base}_eq{eq_id}.csv"
    # Exporting these ruquired data as a csv file
    df_output.to_csv(OUTPUT_DIC / out_name, index=False)

print("files number in match table:", len(df))
file_count = len(list(OUTPUT_DIC.glob("*.csv")))
print("extracted data number:", (file_count))
if missing_file:
    print("Missing orbit files(not found:)")
    pprint.pprint(missing_file)

# 查看轨道数据文件名重复的轨道数据
orbit_list = df["orbit_file"].astype("string").str.strip()
dup = orbit_list[orbit_list.duplicated()]
print("dup_count:", dup.size)
print("dup_samples:", dup.head(20).tolist())

# 查看是否全部地震轨道进行抽取
# check_list = [p.name for p in OUTPUT_DIC.glob("*.csv")]     # OUTPUT_DIC内のファイルをファイル名のみ、文字列として一覧しリストとして出力する。
# pprint.pprint(check_list)
# print(type(check_list[10]))
# match_list = list(df["orbit_file"].astype("string"))
# pprint.pprint(match_list)
# print(type(match_list[10]))

# if check_list == match_list:
#     print("OK")
#     print("total:", len(match_list))
# else:
#     diff = Counter(match_list) - Counter(check_list)  # 足りない分だけ残る
#     print("NG")
#     print("missing_total:", sum(diff.values()))
#     print("missing_items:", dict(diff))
