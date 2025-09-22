import pandas as pd
import numpy as np
import os
import datetime

# 定数
Declustring_Period_Days_Threshold = 30 # デクラスタリング期間[days]
Declustring_Distance_Threshold = 30    # デクラスタリング距離[km]
Input_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Earthquake_catalog\Original\EarthquakeCatalog_Depth40kmbelow_Magnitude4.8above.csv'
Output_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Declustring_30days_30km.csv'
Output_Log_CSV_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Declustring_Log.csv'
Log_File_Path = 'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Declustring\Log.txt'

# 発生時刻の新しい順にソートする。
df_sorted = Input_CSV_Path.sort_values(['time'],ascending=False).copy()
df_sorted['is_aftershock'] = False  # 余震フラグ列を追加

# Gardner&Knopoffの経験則

