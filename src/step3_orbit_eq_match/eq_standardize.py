# -*- coding: utf-8 -*-
# eq_standardize.py
###################################################################################################
###################################################################################################
# Python script to standaradize the earthquake data as required.
# =================================================================================================
# Structure of the script:
#--------------------------------------------------------------------------------------------------
# [1.   Importing the modules]
#
#
# [2.   Setting the path/directory]
#
#
# [3.   Importing inputdata]
# 3-1.  Importint the eq_data as a data frame.
#
#
# [4.   Converting format of "time" column]
# 4-1.  Converting string into datetime64(without timezone info) which is reserved in EQ_DATA
#
# 4-2.  Changing datetime format
#
# 4-3.  Converting the new format "datetime" from object into datetime64
#
# 4-4.  "datetime"列で昇順ソートする。
#
#
# [5.   Modifying the DataFrame as requested]
# 5-1.  Caculating the time before earthquake occurence
#
# 5-2.  Creating "eq_id" as a new column
#
# 5-3.  Changing the order of columns
#
#
# [6.   Exporting the data frame as a csv file]
# 6-1.  Outputing dataframe as a csv file.
#
#
###################################################################################################
# [1.   Importing the modules]
import pandas as pd
import numpy as np
from pathlib import Path
#
#**************************************************************************************************
# [2.   Setting the path/directory]
EQ_DATA = Path(r"E:\tables\earthquake_catalog\declustered\eq_m4.8above_depth40kmbelow_2004_2010_declustered_ver3.csv")
OUTPUT_DIR = Path(r"E:\tables\earthquake_catalog\standardize")
#
#**************************************************************************************************
# [3.   Importing inputdata]
# 3-1.  Importint the eq_data as a data frame.
df = pd.read_csv(EQ_DATA,usecols=["time","latitude","longitude","depth","mag"])
df.info()
print(df)
#
#**************************************************************************************************
# [4.   Converting format of "time" column]
# 4-1.  Converting string into datetime64(without timezone info) which is reserved in EQ_DATA
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True).dt.tz_localize(None)
df.info()
#
# 4-2.  Changing datetime format
df["time"] = df["time"].dt.strftime("%Y/%m/%d  %H:%M:%S")
#
# 4-3.  Converting the new format "datetime" from object into datetime64
df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d  %H:%M:%S")
df["datetime"] = df["time"]
df= df.drop("time", axis=1)
print("Converting the new format time from object into datetime64")
df.info()
print(df)
#
# 4-4.  "datetime"列で昇順ソートする。
df = df.sort_values("datetime")
print("Sorting by the time column in ascending order.")
df.info()
print(df)
#
#**************************************************************************************************
# [5.   Modifying the DataFrame as requested]
# 5-1.  Caculating the time before earthquake occurence
df["4h_before"] = df["datetime"] - pd.Timedelta(hours=4)
print("Calculating datetime of 4hour before occurence")
df.info()
print(df)
#
# 5-2.  Creating "eq_id" as a new column
df = df.reset_index(drop=True)
df = df.reset_index(names="eq_id")
print("Inserting eq_id column")
df.info()
print(df)
#
# 5-3.  Changing the order of columns
df = df.reindex(columns=["eq_id","4h_before","datetime","latitude","longitude","depth","mag"])
print("Changing the order of columns")
df.info()
print(df)
#
#**************************************************************************************************
# [6.   Exporting the data frame as a csv file]
# 6-1.  Outputing dataframe as a csv file.
df.to_csv(OUTPUT_DIR / "eq_standardize_ver4.csv", index=False)
#
#**************************************************************************************************