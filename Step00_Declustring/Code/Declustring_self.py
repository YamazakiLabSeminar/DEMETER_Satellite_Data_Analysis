import pandas as pd
import numpy as np
import os
import datetime
from scipy import interpolate
from geopy.distance import geodesic

# 定数
Declustring_Period_Days_Threshold = 30 # デクラスタリング期間[days]
Declustring_Distance_Threshold = 30    # デクラスタリング距離[km]
Input_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Earthquake_catalog\Original\EarthquakeCatalog_Depth40kmbelow_Magnitude4.8above.csv'
Output_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Declustring_30days_30km.csv'
Output_Log_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Declustring_Log.csv'
Log_File_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Log.txt'

# Gardner&Knopoffの経験則
## ルックアップテーブルの作成
magnitude_bins = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
time_days_bins = [6,11.5,22,42,83,155,290,510,790,915,960,985]
distance_km_bins = [19.5,22.5,26,30,35,40,47,54,61,70,81,94]

## 線形補間関数の作成
time_interp = interpolate.interp1d(magnitude_bins, time_days_bins, kind='linear',fill_value=(time_days_bins[0], time_days_bins[-1]), bounds_error=False)
dist_interp = interpolate.interp1d(magnitude_bins, distance_km_bins, kind='linear',fill_value=(distance_km_bins[0], distance_km_bins[-1]), bounds_error=False)

## 線形補間関数の定義
def get_gk_windows_interp(magnitude):
    time_window = float(time_interp(magnitude))
    dist_window = float(dist_interp(magnitude))
    return time_window, dist_window

############################################################
# デクラスタリング関数の定義
def decrustring():
    ## ログファイルを読み込み、現在のインデックス番号を取得する。
    if os.path.exists(Log_File_Path):
        with open(Log_File_Path, 'r') as log_file:
            log_index = int(log_file.read())
            print(f"Resuming from index {log_index}")
    else:
        log_index = 0
        print("Starting from the beginning")
    
    ## 地震カタログを発生時刻の新しい順にソートする。
    df_sorted = Input_CSV_Path.sort_values(['time'],ascending=False).copy()
    df_sorted['is_aftershock'] = False  # 余震フラグ列を追加

    ## デクラスタリング処理
    for i, mainshock in df_sorted.iterrows():
        if mainshock['is_aftershock']:
            continue  # 既に余震とマークされている場合はスキップ
        
        ###本震の情報を取得
        main_magnitude = mainshock['magnitude']
        main_time = mainshock['time']
        main_lat = mainshock['latitude']
        main_lon = mainshock['longitude']

        ### Gardner&Knopoffの経験則に基づき、メインショックのマグニチュードから時間窓と距離窓を取得
        time_window, dist_window = get_gk_windows_interp(main_magnitude)
        
        ### 余震の検索
        #### 時間窓内の地震を抽出
        time_mask = (df_sorted['time'] >= main_time - time_window) & (df_sorted['time'] <= main_time + time_window) 
        potential_aftershocks = df_sorted[time_mask]
        # プール配列より、時間窓内の地震を抽出し、df_sortedの余震フラグ列をTrueに更新

        #### 時間窓内の地震のSerial（potential_aftershocks）が、本震かつマグニチュードが本震より大きい地震かつ
        for j, event in potential_aftershocks.iterrows():

            ##### 本震、Mが本震より大きい地震は除外
            if i == j or event['is_aftershock'] or event['magnitude'] >= main_magnitude:
                continue

            ##### 距離窓内の地震を抽出
            event_lat = event['latitude']
            event_lon = event['longitude']
            distance_km = geodesic((main_lat, main_lon), (event_lat, event_lon)).km

            if distance_km <= dist_window:
                df_sorted.at[j, 'is_aftershock'] = True  # 余震フラグをTrueに更新
            
        
