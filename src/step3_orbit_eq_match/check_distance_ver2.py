import pandas as pd
from pathlib import Path
import math
from tqdm.auto import tqdm

ORBIT_DATA_DIR  = Path(r"E:\interim\step2_normalized")
EQ_PATH         = Path(r"E:\tables\orbit_earthquake_candidate\orbit_quake_ver6.csv")
OUTPUT_DIR      = Path(r"E:\tables\orbit_eq_match")

# [1. 地球の離心率の計算]
#--------------------------------------------------------------------------------------------------
# WGS84 model
a = 6378137                                     # the long radius of the Earth
b = 6356752.314245                              # the short radius of the Earth

e = math.sqrt((a*a - b*b)/(a*a))                # eccentricity of the Earth

print("a:<type>{},", type(a), a)
print("b:<type>{},", type(b), b)
print("e:<type>{}", type(e), e)
#**************************************************************************************************

# [2.   入力データのインポート]
#--------------------------------------------------------------------------------------------------
# 2-1.  地震カタログ(時間条件に満たした候補軌道番号が付いたやつ)orbit_quake_ver*.csvを読み取る。
df = pd.read_csv(
    EQ_PATH,
    usecols=[
        "eq_id",
        "4h_before",
        "datetime",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "orbit_meet_time_1",
        "orbit_meet_time_2",
        "orbit_meet_time_3",
    ]
)
df.info()
# 候補軌道列に欠損値、空白セルがあれば、その行を消し、インデックスリセットし、元のインデックスを捨てる。
df = df.dropna(subset=["orbit_meet_time_1","orbit_meet_time_2","orbit_meet_time_3"], how="all"
               ).reset_index(drop=True)

df["4hour_before"] = pd.to_datetime(df["4h_before"], format="mixed")
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
print("DataFrame of searchdata\n")
df.info()
print("")
print(df)

# 2-2.  data frameの行数を返す。
length = len(df)

# 2-3.  地震データと地震軌道データを保存するlistを作成する。
list1 = []
used_orbit_files = []
#**************************************************************************************************

# [3.   距離条件に従う軌道の抽出]
#--------------------------------------------------------------------------------------------------
count_dist = 0
candidate_columns = [7, 8, 9]
for i in tqdm(range(length), desc="Searching", unit="eq"):     # dfのデータの行数の範囲
    lat1 = math.radians(df["latitude"].iloc[i])      # 地震の緯度(ラジアン変換)
    lon1 = math.radians(df["longitude"].iloc[i])      # 地震の経度(ラジアン変換)
    start = df["4hour_before"].iloc[i]      # 地震発生4時間前の時刻
    end = df["datetime"].iloc[i]            # 地震発生時刻

    found = False
    for col in candidate_columns:
        orbit_num = df.iat[i, col]
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
                                 lat1, lon1,df["depth"].iloc[i],df["mag"].iloc[i],
                                 orbit_num, s1, lat2, lon2, df_ob["mlat"].iloc[j], df_ob["mlon"].iloc[j],dist,int(j)])
                    found = True
                    break
        if found:
            break


print("dist < 330 count =", count_dist)

data = pd.DataFrame(list1,columns=["eq_id","4h_before","occur_time","eq_lat","eq_lon","depth","mag",
                                   "orbit_file","330_time","lat","lon","mlat","mlon","dist","330_row"])
data.to_csv(OUTPUT_DIR/"orbit_quake_distance_ver13.csv", index=False)

# 保存: 読み取った全ファイルパス一覧
pd.DataFrame({"file_path": used_orbit_files}).to_csv(
    OUTPUT_DIR / "orbit_files_used_ver13.csv", index=False
)
