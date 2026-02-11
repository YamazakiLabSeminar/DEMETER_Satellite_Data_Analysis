import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
# パスの設定
USGS_PATH = Path(r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv")
OUTPUT_DIR = Path(r"E:\tables\earthquake_catalog\declustered")

def apply_utsu_window(magnitude):
    # 宇津の公式(1970)等の基準に基づく窓の設定例
    # D: 距離(km), T: 時間(days)
    d_km = 10** (0.5 * magnitude - 1.8)
    t_days = 10** (0.5 * magnitude - 1.9)
    return d_km, t_days

def calculate_distance(lat1, lon1, lat2, lon2):
    # 地球半径 (km)
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def decluster_utsu(df):
    # マグニチュードの大きい順にソート（大きい地震を本震として優先するため）
    df = df.sort_values('mag', ascending=False).copy()
    df['is_aftershock'] = False
    
    for i in tqdm(range(len(df)), desc="Declustring", unit="eq"):
        if df.iloc[i]['is_aftershock']:
            continue
            
        mainshock = df.iloc[i]
        d_limit, t_limit = apply_utsu_window(mainshock['mag'])
        
        # 本震以降の地震を抽出
        mask = (df.index != mainshock.name) & (~df['is_aftershock'])
        potential_aftershocks = df[mask]
        
        # 距離と時間の条件判定
        for idx, row in potential_aftershocks.iterrows():
            dist = calculate_distance(mainshock['latitude'], mainshock['longitude'], 
                                      row['latitude'], row['longitude'])
            time_diff = abs((row['time'] - mainshock['time']).days)
            
            if dist <= d_limit and time_diff <= t_limit:
                df.at[idx, 'is_aftershock'] = True
                
    return df[df['is_aftershock'] == False]

# 使い方
# df = pd.read_csv('query.csv', parse_dates=['time'])
    # parse_dates=['time'] により time 列が自動的に datetime64 型になります（文字列のままだと時系列処理がしにくいため）
df = pd.read_csv(USGS_PATH, parse_dates=["time"])
df.info()      
# clean_df = decluster_utsu(df)
clean_df = decluster_utsu(df)
df.info()
clean_df.to_csv(OUTPUT_DIR / "eq_m4.8above_depth40kmbelow_2004_2010_declustered_ver3.csv", index=False)
