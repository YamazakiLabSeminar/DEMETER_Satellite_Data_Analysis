import pandas as pd
import numpy as np
import datetime
import os

# ======== 0.パースの設定 ========
INPUT_DATA = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\eq_m4.8above_depth40kmbelow_200407-201012_add_time_row.csv'
OUTPUT_DATA = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\all_eq_declustring_30day_30km.csv'
LOG_FILE = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\logay_30km.txt'

# ======== 1.定数の設定 ========
days_threshold = 30     # 時間ウィンドウ[days]
distance_threshold = 30     # 距離ウインドウ[km]
earth_radius = 6378.140     # 赤道半径[km]
polar_radius = 6356.755     # 極半径[km]

# ======== 2.関数の定義 ========
## ========= 2.1.時間差の計算 =========
def delta_time_in_days(time1, time2):
    time_diff = abs(time1 - time2)
    return time_diff.total_seconds() / 86400

## ======== 2.2.距離差の計算 ========
def delta_distance_in_km(lon_a, lat_a, lon_b, lat_b):
    if lon_a == lon_b and lat_a == lat_b:
        return 0
    F = (earth_radius - polar_radius) / earth_radius    #   扁平率
    rad_lat_a = np.radians(lat_a)
    rad_lon_a = np.radians(lon_a)
    rad_lat_b = np.radians(lat_b)
    rad_lon_b = np.radians(lon_b)
    pa = np.arctan(polar_radius / earth_radius * np.tan(rad_lat_a))
    pb = np.arctan(polar_radius / earth_radius * np.tan(rad_lat_b))
    xx = np.arccos(np.sin(pa) * np.sin(pb) + np.cos(pa) * np.cos(pb) * np.cos(rad_lon_a - rad_lon_b))
    c1 = (np.sin(xx) - xx) * (np.sin(pa) + np.sin(pb))**2 / np.cos(xx / 2)**2
    c2 = (np.sin(xx) + xx) * (np.sin(pa) - np.sin(pb))**2 / np.sin(xx / 2)**2
    dr = F / 8 * (c1 - c2)
    rho = earth_radius * (xx + dr)  # 測地線長
    return rho

## ======== 2.3.時間データをdatetime形式に変換 ========
def convert_to_datetime(df):
    return[
        datetime.datetime(
            year = int(row['year']),
            month = int(row['month']),
            day = int(row['day']),
            hour = int(row['hour']),
            minute = int(row['minute']),
            second = int(row['second'])
        )
        for _, row in df.iterrows()
    ]

## ======== 2.4.デクラスタリング ========
def declustring():
    # 1.ログファイルの読み込み
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as log_file:
            log_index = int(log_file.read())
            print(f'[Info] Resuming from primary_eq_index={log_index} using log file {LOG_FILE}')
    else:
        log_index = 0
        print(f'[Info] No log file found')

    # 2.データの読み込み
    print(f'[Info] Reading input file: {INPUT_DATA}')
    df = pd.read_csv(INPUT_DATA,
                     usecols = ['year', 'month','day','hour','minute','second','latitude','longitude','magnitude']
    )

    # 3.時間列をdatetime形式に
    df['datetime'] = convert_to_datetime(df)

    # 4.ユニックIDの付与
    df['event_id'] = df.index   #0から連番

    # 5.dfの作成(マグニチュード順/時系列順)
    df_mag = df.sort_values(by='magnitude', ascending=False).reset_index(drop=True)
    df_time = df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
