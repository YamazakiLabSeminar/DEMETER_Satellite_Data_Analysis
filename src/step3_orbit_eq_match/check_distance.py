# -*- coding: utf-8 -*-
# check_distance.py
###################################################################################################
###################################################################################################
# Python script to search the orbits which meet distance requirment.
# The distance between orbit and epicenter is caculated with Hubeny formula.
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
#[1.    Importing Modules]
# 1-1.  Importing pandas, pathlib, tqdm.auto, math.
#
#
#[2.  Setting the directory path]
#2-1.  Setting the input and output directory paht.
#2-2.  Show the directory in the terminal.
#
# [3.   Setting the parameters]
# 3-1.  Setting the long radius of the Earth.
# 3-2.  Setting the short radius of the Earth.
# 3-3.  Caculating the eccentricity of the Earth.
#
#
# [4.   Importing the earthquake data from a csv file which contains orbit data metting time 
#       condition]
#4-1.   Importing the earthquake data from a csv file which contains orbit data metting time 
#       condition.
#4-2.   Caculating the length of the data frame object.
#
#
# [5.   Preparation for searching]
# 5-1.  Making a list which contains earthquake and orbit(meeting distance requirement) data.
#
#
# [6.   Searching orbit which meet distance requirement]
# 6-1.      Setting a loop of earthquake data which contains orbit data meeting time condition.
# 6-2.      Making a list which will contain an orbit meeting distance requirement.
# 6-3.      Extracting orbit num(file name) from df.
# 6-4.      Opening the orbit data with orbit num(file name)
# 6-5.      Importing orbit data from a csv file.
# 6-6.      Caculating distance between the orbit and the epicenter.(Setting a loop of orbit data)
# 6-7.      Converting latitude of the epicenter form degeree into radians notation.
# 6-8.      Converting longitude of the epicenter from degerr into radians notation.
# 6-10.     Converting latitude of the orbit from degree into radians notation.
# 6-11.     Converting longitude of the orbit from degree into radians notation.
# 6-13.     Caculating latitude difference between the epicenter and the orbit.
# 6-14.     Caculating longitude difference between the epicenter and the orbit.
# 6-15.     Normalization the longitude from[0,360] to [-180,+180].
# 6-16.     Caculating the average of latitude of the epicenter and the latitude of the orbit.
# 6-17.     Calculating several parameters which are necessary to caculate the distance.
# 6-18.     Calculating the distance between the epicenter and the orbit in km.
# 6-19.     Cheking whether the distance meeting distance requirement.
# 6-20.     Inserting the data into a list object.
#
#
# [7.       Exproting the data as a csv file]
# 7-1.      Converting a list object into a data frame object.
# 7-2.      Defining the columns of data frame.
# 7-3.      Exporting the data frame object as a csv file.
#
#
#**************************************************************************************************
# Structure of this script:
#--------------------------------------------------------------------------------------------------
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import math

ORBIT_DATA_DIR  = Path(r"E:\interim\step2_normalized")
OB_MEET_TIME    = Path(r"E:\tables\orbit_quake_ver1.csv")
OUTPUT_DIR      = Path(r"E:\tables")

print("orbit data directory =", {ORBIT_DATA_DIR})
print("Orbit_meet_time_condion.csv path=", {OB_MEET_TIME})
print("Directory of output", {OUTPUT_DIR})

a = 6378137                                                         # the long radius of the Earth
b = 6356752.314                                                     # the short radius of the Earth

e = math.sqrt((a*a - b*b)/(a*a))                                    # eccentricity of the Earth

df_obmt = pd.read_csv(OB_MEET_TIME)
length_omt = len(df_obmt)

df_obmt["eq_lat_rad"] = df_obmt["eq_lat"] * math.pi / 180
df_obmt["eq_lon_rad"] = df_obmt["eq_lon"] * math.pi / 180

list1 = [[] for k in range(length_omt)]

for i in tqdm(range(length_omt), desc="Searching", unit="eq"):

    list2 = []

    if df_obmt.iloc[i,7] == df_obmt.iloc[i,7]:

        orbit_file_name = df_obmt.iloc[i,7]
        
        orbit_data_path = ORBIT_DATA_DIR / orbit_file_name

        if not orbit_data_path.exists():
            print(f"Missing orbit data file: {orbit_data_path}")
            raise SystemExit(1)

        df_obdata = pd.read_csv(orbit_data_path)
        df_obdata["lat_rad"] = df_obdata["lat"] * math.pi / 180
        df_obdata["lon_rad"] = df_obdata["lon"] * math.pi / 180
        df_obdata["datetime"] = df_obdata["datetime"].astype(str).str.slice(0, 19)
        df_obdata["datetime"] = df_obdata["datetime"].str.replace("-", "").str.replace(":", "").str.replace(" ", "")

        lat1 = df_obmt["eq_lat_rad"].iloc[i]
        lon1 = df_obmt["eq_lon_rad"].iloc[i]
        beforeq = int(df_obmt["4hour_before"].iloc[i])
        starteq = int(df_obmt["datetime"].iloc[i])

            

        for j in range(len(df_obdata)):
            lat2 = df_obdata["lat_rad"].iloc[j]
            lon2 = df_obdata["lon_rad"].iloc[j]

            dif_lat = lat2 - lat1
            dif_lon = lon2 - lon1
            if dif_lon > math.pi:
                dif_lon -= 2*math.pi
            elif dif_lon < -math.pi:
                dif_lon += 2*math.pi
            
            P = (lat1+lat2) / 2                                         # 両点緯度の平均値
            W = math.sqrt((1-e*e * math.sin(P) * math.sin(P)))
            M = (a*(1 - e*e)) / (W * W * W)
            N = a / W

            dist = math.sqrt(dif_lat*dif_lat*M*M + dif_lon*dif_lon*N*N*math.cos(P)*math.cos(P))
            dist = dist / 1000                                          # 単位[m] => [km]

            if dist < 330:
                s1 = int(df_obdata["datetime"].iloc[j])
                if beforeq <= s1 <= starteq:
                    list2.append(orbit_file_name)
                    break
        
        list1[i] = list2

    if df_obmt.iloc[i,8] == df_obmt.iloc[i,8]:

        orbit_file_name = df_obmt.iloc[i,8]
        
        orbit_data_path = ORBIT_DATA_DIR / orbit_file_name

        if not orbit_data_path.exists():
            print(f"Missing orbit data file: {orbit_data_path}")
            raise SystemExit(1)

        df_obdata = pd.read_csv(orbit_data_path)
        df_obdata["lat_rad"] = df_obdata["lat"] * math.pi / 180
        df_obdata["lon_rad"] = df_obdata["lon"] * math.pi / 180
        df_obdata["datetime"] = df_obdata["datetime"].astype(str).str.slice(0, 19)
        df_obdata["datetime"] = df_obdata["datetime"].str.replace("-", "").str.replace(":", "").str.replace(" ", "")

        lat1 = df_obmt["eq_lat_rad"].iloc[i]
        lon1 = df_obmt["eq_lon_rad"].iloc[i]
        beforeq = int(df_obmt["4hour_before"].iloc[i])
        starteq = int(df_obmt["datetime"].iloc[i])

            

        for j in range(len(df_obdata)):
            lat2 = df_obdata["lat_rad"].iloc[j]
            lon2 = df_obdata["lon_rad"].iloc[j]

            dif_lat = lat2 - lat1
            dif_lon = lon2 - lon1
            if dif_lon > math.pi:
                dif_lon -= 2*math.pi
            elif dif_lon < -math.pi:
                dif_lon += 2*math.pi
            
            P = (lat1+lat2) / 2                                         # 両点緯度の平均値
            W = math.sqrt((1-e*e * math.sin(P) * math.sin(P)))
            M = (a*(1 - e*e)) / (W * W * W)
            N = a / W

            dist = math.sqrt(dif_lat*dif_lat*M*M + dif_lon*dif_lon*N*N*math.cos(P)*math.cos(P))
            dist = dist / 1000                                          # 単位[m] => [km]

            if dist < 330:
                s1 = int(df_obdata["datetime"].iloc[j])
                if beforeq <= s1 <= starteq:
                    list2.append(orbit_file_name)
                    break
        
        list1[i] = list2
       
    if df_obmt.iloc[i,9] == df_obmt.iloc[i,9]:

        orbit_file_name = df_obmt.iloc[i,9]
        
        orbit_data_path = ORBIT_DATA_DIR / orbit_file_name

        if not orbit_data_path.exists():
            print(f"Missing orbit data file: {orbit_data_path}")
            raise SystemExit(1)

        df_obdata = pd.read_csv(orbit_data_path)
        df_obdata["lat_rad"] = df_obdata["lat"] * math.pi / 180
        df_obdata["lon_rad"] = df_obdata["lon"] * math.pi / 180
        df_obdata["datetime"] = df_obdata["datetime"].astype(str).str.slice(0, 19)
        df_obdata["datetime"] = df_obdata["datetime"].str.replace("-", "").str.replace(":", "").str.replace(" ", "")

        lat1 = df_obmt["eq_lat_rad"].iloc[i]
        lon1 = df_obmt["eq_lon_rad"].iloc[i]
        beforeq = int(df_obmt["4hour_before"].iloc[i])
        starteq = int(df_obmt["datetime"].iloc[i])

            

        for j in range(len(df_obdata)):
            lat2 = df_obdata["lat_rad"].iloc[j]
            lon2 = df_obdata["lon_rad"].iloc[j]

            dif_lat = lat2 - lat1
            dif_lon = lon2 - lon1
            if dif_lon > math.pi:
                dif_lon -= 2*math.pi
            elif dif_lon < -math.pi:
                dif_lon += 2*math.pi
            
            P = (lat1+lat2) / 2                                         # 両点緯度の平均値
            W = math.sqrt((1-e*e * math.sin(P) * math.sin(P)))
            M = (a*(1 - e*e)) / (W * W * W)
            N = a / W

            dist = math.sqrt(dif_lat*dif_lat*M*M + dif_lon*dif_lon*N*N*math.cos(P)*math.cos(P))
            dist = dist / 1000                                          # 単位[m] => [km]

            if dist < 330:
                s1 = int(df_obdata["datetime"].iloc[j])
                if beforeq <= s1 <= starteq:
                    list2.append(orbit_file_name)
                    break
        
        list1[i] = list2


data = pd.DataFrame(list1, columns=["orbit_meet_time_dist"])
output_one = pd.concat([df_obmt,data], axis=1)
output_one.to_csv(OUTPUT_DIR/"orbit_quake_distance_ver2.csv", index=False)
#**************************************************************************************************
