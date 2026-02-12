# -*- coding: utf-8 -*-
# extract_required_data.py
###################################################################################################
###################################################################################################
# Python script to calculate the distance between sample and epicenter of the individual sea 
# analysis candidate orbit file.
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
MAT_PATH = Path(r"E:\tables\orbit_eq_match\orbit_quake_distance_ver17.csv")
CAND_DIC = Path(r"E:\interim\orbit_data_for_sea_analysis_candidate")
OUTPUT_DIC = Path(r"E:\interim\orbit_data_for_sea_analysis_candidate_dist")
#
# 2-2.  パス存在しない場合の対策
OUTPUT_DIC.mkdir(parents=True, exist_ok=True)
#
#
# [3.   Importing the table of earthquake-orbits matching]
# 3-1.  Importing the talbe of earthquake-orbits macthing
df = pd.read_csv(MAT_PATH)
df["orbit_file"] = df["orbit_file"].astype("string")

df.info()
print(df)
#
#
a = 6378137                                     # the long radius of the Earth
b = 6356752.314245                              # the short radius of the Earth

e = math.sqrt((a*a - b*b)/(a*a))                # eccentricity of the Earth
# [4.   Caculating the distance between sample and epicenter of the individual sea 
# analysis candidate orbit file.]
# Creating a list which contains the whole file name in sea analysis candidate folder.
# cand_list = [p.name for p in CAND_DIC.glob("*.csv")]
# pprint.pprint(cand_list)
# print(type(cand_list[10]))
imported_count = 0
# Matching Tableの行ずつ読み取るループを作成する。
for i in tqdm(range(len(df)), desc="calculate dist", unit="file"):
    # 震央緯度、経度を抽出する。
    lat1 = df["eq_lat"].iloc[i]
    lon1 =df["eq_lon"].iloc[i]
    # print(type(lon1), type(lat1))

    # ファイル名を抽出する。
    file_name = df["orbit_file"].iloc[i]
    # マッチング用ファイル名を作成する。
    eq_id = df["eq_id"].iloc[i]
    stem = Path(file_name).stem  
    base = stem.split("_step")[0]
    file_name = f"{base}_eq{eq_id}.csv"
    # print(file_name, type(file_name))

    # CAND_DICにある同名ファイルを読みとる。
    df_ob = pd.read_csv(CAND_DIC / file_name)
    list1=[]
    imported_count += 1
    # df_ob.info()
    for j in range(len(df_ob)):
        lat2 = math.radians(df_ob["lat"].iloc[j])
        lon2 = math.radians(df_ob["lon"].iloc[j])
        # print(lat2, type(lat2), lon2, type(lon2))
        dif_lat = lat2 - lat1
        dif_lon = lon2 - lon1
        
        if dif_lon > math.pi:
            dif_lon = 2*math.pi - dif_lon

        P = (lat1+lat2) / 2.0                                         # 両点緯度の平均値
        W = math.sqrt(1.0-e*e * math.sin(P) * math.sin(P))
        M = (a*(1.0 - e*e)) / (W * W * W)
        N = a / W

        t1 = M * dif_lat
        t2 = N * dif_lon * math.cos(P)

        dist = math.sqrt(t1 ** 2 + t2 ** 2)
        dist = dist / 1000.0                                          # 単位[m] => [km]

        list1.append(dist)
    # Exporting as a csv file.
    df_ob["dist"] = list1
    df_ob.to_csv(OUTPUT_DIC / file_name, index=True)

print("total candidate num:", {len(df)})
print("imported_count:", imported_count)