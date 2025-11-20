import pandas as pd
import numpy as np
import os
import datetime

SSD_DIRECTORY = r'F:/'
SAT_DATA_DIR = r'F:/01_EFdata'
TIME_SEPERATED_2230LT = r'F:/02_Time_Seperated_2230LT'
TIME_SEPERATED_4HOUR = r'F:/03_Time_Seperated_4hour'
DISTANCE_330KM = r'F:/04_Distance_330km'
OUTPUT_NIGHT_ORBIT_LIST = r'F:/night_orbit_list.txt'
print(f'[Info] 現在のディレクトリ:{os.getcwd()}')

# DirをSSDにある衛星データに切り替える
os.chdir(SAT_DATA_DIR)
if not os.path.exists(SAT_DATA_DIR):
    print(f'[Info] ディレクトリが見つかりません:{SAT_DATA_DIR}')
else:
    print(f'[Info] ディレクトリを切り替えました:{os.getcwd()}')

# 夜間軌道リスト
night_orbit_list=[]
night_time_start = datetime.time(22,30)
night_time_end = datetime.time(6,0)

print("[Info] 処理開始...")

# folder内のファイル名を読み取り、軌道番号(昇順)にソートする
for filename in sorted(os.listdir(SAT_DATA_DIR)):
    if not filename.endswith('.csv'):
        continue
    if filename.startswith('.'): 
        continue
    filepath = os.path.join(SAT_DATA_DIR, filename)

    # ファイル名を分解する(例えば、DMT_N1_1132_00197.1.csv)
    parts = filename.replace(".csv", "").split("_") # 拡張子を除去、"_"で分解
    if len(parts) < 4: # ファイル名の長さが4以下の場合
        continue
    orbit_part = parts[3] # ファイル名の3番目の要素を取り出す
    orbit_num = orbit_part.split(".")[0] # ".1"や".0"などを除去する

    try:
        df = pd.read_csv(filepath)

        cols = ["year","month","date","hour","min","sec","milsec"]

        # --- 数字以外の行を NaN にする ---
        for c in cols:
            df[c] = df[c].astype(str).str.strip()

            # 数字以外（マイナスや小数も許可）を判別
            mask = df[c].str.match(r'^-?\d+(\.\d+)?$')
            df.loc[~mask, c] = None

        # --- float → int へ（小数は切り捨て）---
        df[cols] = df[cols].astype(float).astype(int)

        # --- microsecond へ変換 ---
        df["datetime"] = pd.to_datetime(
            dict(
                year=df["year"],
                month=df["month"],
                day=df["date"],
                hour=df["hour"],
                minute=df["min"],
                second=df["sec"],
                microsecond=df["milsec"] * 1000
            ),
            errors="coerce"
        )

        # --- datetime 生成できなかった行は削除 ---
        df = df.dropna(subset=["datetime"])

        if df.empty:
            continue
        # --- 最初の時刻を取得 ---
        first_time = df["datetime"].iloc[0].time()

        if (first_time >= night_time_start) or (first_time <= night_time_end):
            night_orbit_list.append(orbit_num.zfill(5))
        else:
            continue
        
    except Exception as e:
        print(f'[Error] 読み込み失敗{filename}:{e}')
    
with open(OUTPUT_NIGHT_ORBIT_LIST, 'w', encoding='utf-8')as f:
    f.write(f'#DEMETER夜間衛星軌道リスト(2230LT~0600LT)\n')
    f.write(f"軌道総数:{len(night_orbit_list)}\n")
    for orbit in night_orbit_list:
        f.write(orbit+"\n")

print(f"\n============完了===============")
print(f"夜間軌道数:{len(night_orbit_list)}")
print(f"結果を保存しました:{OUTPUT_NIGHT_ORBIT_LIST}")