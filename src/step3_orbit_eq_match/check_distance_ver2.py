import pandas as pd
from pathlib import Path
import math
from tqdm.auto import tqdm

ORBIT_DATA_DIR  = Path(r"E:\interim\step2_normalized")
EQ_PATH         = Path(r"E:\tables\orbit_earthquake_candidate\orbit_quake_ver11.csv")
OUTPUT_DIR      = Path(r"E:\tables\orbit_eq_match")

# [2.   入力データのインポート]
# 2-1.  地震カタログ(時間条件に満たした候補軌道番号が付いたやつ)orbit_quake_ver*.csvを読み取る。
df = pd.read_csv(EQ_PATH)
df.info()
#
#
# [3.   Preparing for cheking the distance between sample and epicenter]
# 3-1.  離心率の計算
# WGS84 model
a = 6378137                                     # the long radius of the Earth
b = 6356752.314245                              # the short radius of the Earth

e = math.sqrt((a*a - b*b)/(a*a))                # eccentricity of the Earth

print("a:<type>{},", type(a), a)
print("b:<type>{},", type(b), b)
print("e:<type>{}", type(e), e)
#
# 3-2.  地震データと地震軌道データを保存するlistを作成する。
length = len(df)
list1 = []
used_orbit_files = []
#
# 3-3.  時間条件候補リストの列名（列位置固定を避ける）
candidate_columns = [c for c in df.columns if c.startswith("orbit_meet_time_")]
if not candidate_columns:
    raise ValueError("No orbit candidate columns found (expected columns like orbit_meet_time_*)")
#
# 3-4.  距離条件に従う軌道の数をカウントする変数
count_dist = 0
#
#
# [4.   距離条件に従う軌道の抽出]
# 前処理（ループ内で同じ変換を繰り返さない）
df["4h_before"] = pd.to_datetime(df["4h_before"], format="mixed", errors="coerce")
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")

# 4-1.  地震データの行数の範囲でループを作成する。
for i in tqdm(range(length), desc="Searching", unit="eq"):     # dfのデータの行数の範囲
#
# 4-2.  地震の緯度、経度(0-360)、地震発生4時間前の時刻、地震発生時刻を抽出する。
    lat1 = math.radians(df["latitude"].iloc[i])      # 地震の緯度(ラジアン変換)
    lon1 = math.radians(df["longitude_360"].iloc[i])      # 地震の経度(ラジアン変換)
    start = df["4h_before"].iloc[i]      # 地震発生4時間前の時刻
    end = df["datetime"].iloc[i]         # 地震発生時刻

    for col in candidate_columns:
        orbit_num = df.at[i, col]
        if pd.isna(orbit_num):
            continue
        orbit_num = str(orbit_num).strip()
        if orbit_num.endswith(".0") and orbit_num[:-2].isdigit():
            orbit_num = orbit_num[:-2]
        if not orbit_num.lower().endswith(".csv"):
            orbit_num = f"{orbit_num}.csv"

        file_path = ORBIT_DATA_DIR / orbit_num
        if not file_path.exists():
            base = orbit_num[:-4] if orbit_num.lower().endswith(".csv") else orbit_num
            matches = sorted(ORBIT_DATA_DIR.glob(f"{base}*.csv"))
            if len(matches) == 1:
                file_path = matches[0]
            elif len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple orbit files match prefix '{base}': {matches}"
                )
            else:
                raise FileNotFoundError(f"Orbit file not found: {file_path}")
        used_orbit_files.append(str(file_path))
        df_ob = pd.read_csv(file_path, usecols=["datetime","lat","lon","mlat","mlon"])
        df_ob["datetime"] = pd.to_datetime(df_ob["datetime"], format="mixed")

        for j in range(len(df_ob)):     # df_obのデータの行数の範囲
            lat2 = math.radians(df_ob["lat"].iloc[j])   # 軌道データで各サンプルの緯度(ラジアン変換)
            lat2 = float("{:.8f}".format(lat2))                # 軌道データ緯度小数点以下8桁
            lon2 = math.radians(df_ob["lon"].iloc[j])   # 軌道データで各サンプルの経度(ラジアン変換)
            lon2 = float("{:.7f}".format(lon2))                # 軌道データ経度小数点以下7桁
            
            dif_lat = lat2 - lat1
            dif_lon = lon2 - lon1

            if dif_lon > math.pi:
                dif_lon = 2*math.pi - dif_lon

            if dif_lon < -math.pi:
                dif_lon = 2*math.pi + dif_lon
            P = (lat1+lat2) / 2.0                                         # 両点緯度の平均値
            W = math.sqrt(1.0-e*e * math.sin(P) * math.sin(P))
            M = (a*(1.0 - e*e)) / (W * W * W)
            N = a / W

            t1 = M * dif_lat
            t2 = N * dif_lon * math.cos(P)

            dist = math.sqrt(t1 ** 2 + t2 ** 2)
            dist = dist / 1000.0                                          # 単位[m] => [km]

            if dist < 330.0:
                s1 = df_ob["datetime"].iloc[j]
                if start < s1 < end:
                    count_dist += 1
                    list1.append([df["eq_id"].iloc[i],
                                 start,
                                 end,
                                 df["latitude"].iloc[i], df["longitude"].iloc[i],
                                 df["longitude_360"].iloc[i],df["mag"].iloc[i],
                                 orbit_num, s1, df_ob["lat"].iloc[j], df_ob["lon"].iloc[j],
                                 df_ob["mlat"].iloc[j], df_ob["mlon"].iloc[j],dist,int(j)])
                    break


print("dist < 330 count =", count_dist)

data = pd.DataFrame(list1,columns=["eq_id","4h_before","occur_time","latitude","longitude","longitude_360","mag",
                                   "orbit_file","330_time","lat","lon","mlat","mlon","dist","330_row"])
data.to_csv(OUTPUT_DIR/"orbit_quake_distance_ver19.csv", index=False)

# 保存: 読み取った全ファイルパス一覧
pd.DataFrame({"file_path": used_orbit_files}).to_csv(
    OUTPUT_DIR / "orbit_files_used_ver19.csv", index=False
)
