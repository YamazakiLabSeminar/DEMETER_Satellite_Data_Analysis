import pandas as pd
import numpy as np
import os
import datetime

# 定数設定
TIME_WINDOW_DAYS = 30  # デクラスタリング期間の閾値[days]
DISTANCE_WINDOW = 30  # デクラスタリング距離の閾値[km]
EARTH_RADIUS = 6378.140  # 地球の半径[km]
POLAR_RADIUS = 6356.755  # 極の半径[km]

# 二つ地震の時間差を計算する関数
def calculate_delta_time_in_days(time1, time2):
    delta_time = abs(time1 - time2)
    return delta_time.total_seconds()/86400  # 秒を日に変換

# 二つ地震の距離差を計算する関数
def calculate_distance(lon_a, lat_a, lon_b, lat_b):
    """2地点の地球上の距離を計算"""
    if lon_a == lon_b and lat_a == lat_b:
        return 0
    F = (EARTH_RADIUS - POLAR_RADIUS) / EARTH_RADIUS  # 扁平率
    rad_lat_a = np.radians(lat_a)
    rad_lon_a = np.radians(lon_a)
    rad_lat_b = np.radians(lat_b)
    rad_lon_b = np.radians(lon_b)
    pa = np.arctan(POLAR_RADIUS / EARTH_RADIUS * np.tan(rad_lat_a))
    pb = np.arctan(POLAR_RADIUS / EARTH_RADIUS * np.tan(rad_lat_b))
    xx = np.arccos(np.sin(pa) * np.sin(pb) + np.cos(pa) * np.cos(pb) * np.cos(rad_lon_a - rad_lon_b))
    c1 = (np.sin(xx) - xx) * (np.sin(pa) + np.sin(pb))**2 / np.cos(xx / 2)**2
    c2 = (np.sin(xx) + xx) * (np.sin(pa) - np.sin(pb))**2 / np.sin(xx / 2)**2
    dr = F / 8 * (c1 - c2)
    rho = EARTH_RADIUS * (xx + dr)  # 測地線長
    return rho

# 時間データをdatetime形式に変換する関数
def convert_to_datetime(df):
    """データフレームの日時情報をdatetime形式に変換"""
    return [
        datetime.datetime(
            year=int(row['year']),
            month=int(row['month']),
            day=int(row['day']),
            hour=int(row['hour']),
            minute=int(row['minute']),
            second=int(row['second'])
        )
        for _, row in df.iterrows()
    ]

# デクラスタリング関数の定義
def declustering():

    # === 0.ディレクトリ設定 ===
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(current_dir, 'Data', 'EarthquakeCatalog_Depth40kmbelow_Magnitude4.8above_TimeSeperate.csv')

    # ファイル名を動的に生成: 例: all-eq-declustering-30day-30km.csv
    output_file_name = f'all-eq-declustering-{TIME_WINDOW_DAYS}day-{DISTANCE_WINDOW}km.csv'
    output_log_csv_name = f'all-eq-declustering-log-{TIME_WINDOW_DAYS}day-{DISTANCE_WINDOW}km.csv'
    log_file_name = f'log-{TIME_WINDOW_DAYS}day-{DISTANCE_WINDOW}km.txt'

    #ファイルパスを設定
    output_csv_path = os.path.join(current_dir, 'Data', output_file_name)
    output_log_csv_path = os.path.join(current_dir, 'Data', output_log_csv_name)
    log_file_path = os.path.join(current_dir, 'Data', log_file_name)

    # === 1.ログファイルを読み込み ===
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_index = int(log_file.read())
            print(log_index)
    else:
        log_index = 0
        print(f"[Info] Resuming from primary_eq_index={log_index} using log file {log_file_path}")

    # === 2.地震データを読み込み ===
    print(f'[Info] Reading input CSV: {input_csv_path}')
    df = pd.read_csv(input_csv_path,usecols=['year','month','day','hour','minute','second','latitude','longitude','magnitude'])

    # === 3.日時データをdatetime形式に変換 ===
    df['datetime'] = convert_to_datetime(df)

    # === 4. ユニックIDを与える ===
    df.sort_values(by='datetime',ascending=True,inplace=True) # 日時に基づいて、古いから新しい順にソート
    df.reset_index(drop=True,inplace=True) # インデックスを振り直す
    df['event_id'] = df.index  # ユニックIDを与える

    # === 5.二つDataFrameを作成する(マグニチュード降順/時系列昇順) ===
    df_magnitude = df.sort_values(by='magnitude',ascending=False).reset_index(drop=True) # マグニチュード降順
    df_time = df.sort_values(by='datetime',ascending=True).reset_index(drop=True) # 時系列昇順
    # 削除フラグ列を作成
    N = len(df)
    remove_flags = np.zeros(N, dtype=bool)  # Falseで初期化

    # === 6.デクラスタリング処理:マグニチュードの順に走査 ===
    total_events = len(df_magnitude)
    for primary_eq_index, primary_eq in df_magnitude.iterrows():
        # ログファイルのインデックス以降ならばスキップ
        if primary_eq_index < log_index:
           continue
       
        e_id = primary_eq['event_id']
       # すでに削除されている場合はスキップ
        if remove_flags[e_id]:
           continue
        primary_eq_time = primary_eq['datetime']

        # 時間的閾値を超える時刻
        cutoff_time = primary_eq_time + datetime.timedelta(days=TIME_WINDOW_DAYS)

        # 時系列DataFrameで[primary_eq_time < time <= cutoff_time]を満たす地震を抽出
        start_idx = df_time['datetime'].searchsorted(primary_eq_time, side='right')
        end_idx = df_time['datetime'].searchsorted(cutoff_time, side='left')

        for j in range(start_idx, end_idx):
            secondary_eq_id = df_time.loc[j, 'event_id']
            # すでに削除されている場合はスキップ
            if remove_flags[secondary_eq_id]:
                continue

            secondary_eq_time = df_time.loc[j, 'datetime']
            # 時間差の確認
            if calculate_delta_time_in_days(primary_eq_time, secondary_eq_time) > TIME_WINDOW_DAYS:
                continue
            
            # 距離差の確認
            lon_a, lat_a = primary_eq['longitude'], primary_eq['latitude']
            lon_b, lat_b = df_time.loc[j, 'longitude'], df_time.loc[j, 'latitude']
            delta_distance = calculate_distance(lon_a, lat_a, lon_b, lat_b)
            if delta_distance <= DISTANCE_WINDOW:
                remove_flags[secondary_eq_id] = True  # 削除フラグを立てる
        
        # 500イベントごとに中間保存
        if primary_eq_index % 500 == 0:
            print(f"[Progress] primary_eq_index={primary_eq_index}/{total_events}")
            tmp_remaining = df[~remove_flags]
            tmp_remaining.to_csv(output_csv_path, index=False)
            with open(log_file_path, 'w') as log_file:
                log_file.write(str(primary_eq_index))

    remaining_data = df[~remove_flags].copy()
    remaining_data.to_csv(output_csv_path, index=False)
    print(f'[Info] Declustering completed. Output saved to {output_csv_path}')

    if __name__ == "__main__":
        declustering()
