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
#
#
#[4.  Preparation of searching]
#4-1. Caculate the length of earthquake data
length_eqdata = len(eq_data)
#
#4-2.  Making a list. This will contains orbit data which meet time requirement.
list1 = []
#
#
#[5.  Searching orbit which meet time requirement]
for i in range(length_eqdata):
#5-1.  Setting time four hour before the earthquake occurred.
    beforeq = pd.to_datetime(eq_data["4hour_before"].iloc[i])
    starteq = pd.to_datetime(eq_data["datetime"].iloc[i])
#
#5-2.  Conversing the time into int format
    beforeq = int(beforeq.strftime("%Y%m%d%H%M%S"))
    starteq = int(starteq.strftime("%Y%m%d%H%M%S"))
#
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
            list1.append(searchdata.iloc[j,0])
        elif beforeq <= e1 <= starteq:
            list1.append(searchdata.iloc[j,0])
#
#
#[6.  Exporting data as a csv format file]
#6-1.  converting a list object into a data frame object.
data = pd.DataFrame(list1)
#
#6-2.  Connecting orbit data and earthquake data.
outputone = pd.concat([eq_data, data], axis=1)
#
#6-3.  Exporting the data frame into a csv file.
outputone.to_csv(OUTPUT_DIR / "orbit_meet_time_condition.csv")
###################################################################################################
