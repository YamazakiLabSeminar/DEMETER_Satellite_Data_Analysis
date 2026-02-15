# -*- coding: utf-8 -*-
# eq_standardize.py
###################################################################################################
###################################################################################################
# Python script to removed the aftershocks from earthquake catalog by using declustering method.
# =================================================================================================
# Structure of the script:
#--------------------------------------------------------------------------------------------------
# [1.   Importing the modules]
# 1-1.  Importing pandas, numpy, and pathlib modules.
import math
from pathlib import Path
import pandas as pd
#
# [2.   Setting the path/directory]
# 2-1.  Setting the path of earthquake catalog and output directory.
EQ_DATA = Path(r"E:\tables\earthquake_catalog\preprocessed\eq_m4.8above_depth40kmbelow_2004-2010_preprocessed.csv")
OUTPUT_DIR = Path(r"E:\tables\earthquake_catalog\declustered")
# 2-2.  出力ディレクトリが存在しない場合は作成する。
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# [3.   Importing inputdata]    
# 3-1.  Setting the usecolumns.
USECOLS = ["eq_id","4h_before","datetime","latitude","longitude","longitude_360","depth","mag"]
#
# 3-2.  Importing earthquake data as a data frame.
df = pd.read_csv(EQ_DATA, usecols=USECOLS)
#
#
# [4.   Preparing for declustering]
# 4-1.  Setting the threshold for declustering.
DAYS_THRESHOLD = 30     # aftershockとみなす時間の閾値（日）
DISTANCE_THRESHOLD = 30  # aftershockとみなす距離の閾値（km）
#
# 4-2.  離心率の計算
'''
WGS84 model
'''
a = 6378137                                     # the long radius of the Earth
b = 6356752.314245                              # the short radius of the Earth

e = math.sqrt((a*a - b*b)/(a*a))                # eccentricity of the Earth
#
# 4-3.  2点間の距離を計算する関数を定義する。
def calculate_distance(lat1, lon1, lat2, lon2):
    # 緯度と経度をラジアンに変換する
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 緯度の平均を計算する
    P = (lat1_rad + lat2_rad) / 2.0

    # 緯度と経度の差を計算する
    dif_lat = lat1_rad - lat2_rad
    dif_lon = lon2_rad - lon1_rad

    if dif_lon > math.pi:
        dif_lon = 2*math.pi - dif_lon
    elif dif_lon < -math.pi:
        dif_lon = 2*math.pi + dif_lon

    W = math.sqrt(1.0-e*e * math.sin(P) * math.sin(P))
    M = (a*(1.0 - e*e)) / (W * W * W)
    N = a / W

    t1 = M * dif_lat
    t2 = N * dif_lon * math.cos(P)

    dist = math.sqrt(t1 ** 2 + t2 ** 2)
    dist = dist / 1000.0        # 単位[m] => [km]

    return dist
#
#
# [5.   De-clustering the earthquake catalog]
# 5-1.  Sorting the earthquake data frame by "datetime" column in ascending order.
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
df_time = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
#
# 5-2.  Sorting the earthquake data frame by "mag" column in descending order.
df_mag = df.sort_values(by="mag", ascending=False).reset_index(drop=True)
#
# 5-3.  最大震度から順に地震データをループして、aftershockとみなされる地震を削除する。
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")

dt_series = df_time["datetime"]
ID_COL = "eq_id"
removed_ids = set()

for _, primary_eq in df_mag.iterrows():
    primary_event_id = int(primary_eq[ID_COL])
    if primary_event_id in removed_ids:
        continue

    t_a = primary_eq["datetime"]
    cutoff_time = t_a + pd.Timedelta(days=DAYS_THRESHOLD)

    start_idx = int(dt_series.searchsorted(t_a, side="right"))
    end_idx = int(dt_series.searchsorted(cutoff_time, side="right"))

    lat_a = float(primary_eq["latitude"])
    lon_a = float(primary_eq["longitude_360"])

    for j in range(start_idx, end_idx):
        sec_event_id = int(df_time.at[j, ID_COL])
        if sec_event_id in removed_ids:
            continue
        
        if sec_event_id == primary_event_id:
            continue

        if float(df_time.at[j, "mag"]) > float(primary_eq["mag"]):
            continue
        t_b = df_time.at[j, "datetime"]
        if (t_b - t_a) > pd.Timedelta(days=DAYS_THRESHOLD):
            continue

        lat_b = float(df_time.at[j, "latitude"])
        lon_b = float(df_time.at[j, "longitude_360"])
        dist = calculate_distance(lat_a, lon_a, lat_b, lon_b)
        if dist <= DISTANCE_THRESHOLD:
            removed_ids.add(sec_event_id)

remaining_data = df[~df[ID_COL].isin(removed_ids)].copy()
remaining_data = remaining_data.sort_values("datetime").reset_index(drop=True)
remaining_data["datetime"] = pd.to_datetime(remaining_data["datetime"]).dt.strftime("%Y/%m/%d %H:%M:%S")

out_name = f"all-eq-declustering-{DAYS_THRESHOLD}day-{DISTANCE_THRESHOLD}km.csv"
out_path = OUTPUT_DIR / out_name
remaining_data.to_csv(out_path, index=False)

print(f"[Info] Input events: {len(df)}")
print(f"[Info] Remaining events after declustering: {len(remaining_data)}")
print(f"[Info] Saved: {out_path}")
