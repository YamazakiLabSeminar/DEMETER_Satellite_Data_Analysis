# -*- coding: utf-8 -*-
# orbit_start_end.py
###################################################################################################
###################################################################################################
# Python script to search the orbits which meet time requirment
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
#
#**************************************************************************************************
# Structure of this script:
#--------------------------------------------------------------------------------------------------
#[1.  Importing modules]
#1-1.  Importing pandas, and pathlib modules.
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
#
#
#[2.  Setting the directory path]
#2-1.  Setting the input and output directory paht.
ORBIT_PATH = Path(r"E:\tables\orbit_start_end.csv")
EQ_PATH    = Path(r"E:\tables\eq_m4.8above_depth40kmbelow_2004-2010_declustred_standardized.csv")
OUTPUT_DIR = Path(r"E:\tables")
#
#2-2.  Show the directory in the terminal.
print("Orbit_start_end.py file paht=", {ORBIT_PATH})
print("Earthquake catalogue path=", {EQ_PATH})
print("Directory of output", {OUTPUT_DIR})
#
#
#[3.  Importing the orbit data from orbit_start_end.csv]
#3-1.  Importing the orbit data from orbit_start_end.csv
searchdata = pd.read_csv(ORBIT_PATH)
#
#3-2.  Importing the earthquak data from a file
eq_data = pd.read_csv(EQ_PATH)
eq_data["4hour_before"] = eq_data["4hour_before"].astype(str).str.slice(0, 19)
eq_data["4hour_before"] = eq_data["4hour_before"].str.replace("-", "").str.replace(":", "").str.replace(" ", "")

eq_data["datetime"] = eq_data["datetime"].astype(str).str.slice(0, 19)
eq_data["datetime"] = eq_data["datetime"].str.replace("-", "").str.replace(":", "").str.replace(" ", "")

#
#[4.  Preparation of searching]
#4-1. Caculate the length of earthquake data
length_eqdata = len(eq_data)
#
#4-2.  Making a list. This will contains orbit data which meet time requirement.
list1 = [[] for k in range(length_eqdata)]
#
#
#[5.  Searching orbit which meet time requirement]
for i in tqdm(range(length_eqdata), desc="EQ", unit="eq"):
    beforeq = int(eq_data.iloc[i,1])
    starteq = int(eq_data.iloc[i,2])

    list2 = []

#5-3.  Searching orbits which meet time requiremnt.
    for j in range(len(searchdata)):
#        
#5-4.  Setting time when the orbit started and ended.
        s1 = searchdata["start_time"].iloc[j]
        e1 = searchdata["end_time"].iloc[j]   
#
#5-5.  Extracting the orbit which meets time requiremet.
        if starteq < s1:
            break
        elif beforeq <= s1 <= starteq:
            list2.append(searchdata.iloc[j,0])
        elif beforeq <= e1 <= starteq:
            list2.append(searchdata.iloc[j,0])
        
    list1[i] = list2
#
#
#[6.  Exporting data as a csv format file]
#6-1.  converting a list object into a data frame object.
max_cols = max((len(row) for row in list1), default=0)
col_names = [f"orbit_meet_time_{i+1}" for i in range(max_cols)]
data = pd.DataFrame(list1, columns=col_names)
outputone = pd.concat([eq_data, data], axis=1)
#6-3.  Exporting the data frame into a csv file.
outputone.to_csv(OUTPUT_DIR / "orbit_quake_ver1.csv", index=False)
###################################################################################################
