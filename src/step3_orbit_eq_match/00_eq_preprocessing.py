# -*- coding: utf-8 -*-
# eq_preprocessing.py
###################################################################################################
###################################################################################################
# Python script to modify the earthquake catalog data.
# =================================================================================================
# Structure of the script:
#--------------------------------------------------------------------------------------------------
# [1.   Importing the modules]
# 1-1.  Importing pandas, numpy, and pathlib modules.
import math
from pathlib import Path
import pandas as pd
#
#
# [2.   Setting the path/directory]
# 2-1.  Setting the path of earthquake catalog and output directory.
EQ_DATA = Path(r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv")
OUTPUT_DIR = Path(r"E:\tables\earthquake_catalog\preprocessed")
#
# 2-2.  出力ディレクトリが存在しない場合は作成する。
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#
# [3.   Importing inputdata]    
# 3-1.  Setting the usecolumns.
USECOLS = ["time","latitude","longitude","depth","mag"]
#
# 3-2.  Importing earthquake data as a data frame.
df = pd.read_csv(EQ_DATA, usecols=USECOLS)
df.info()
#
# 3-3.  Creating a output data frame
df_output = pd.DataFrame()
#
#
# [4.   Modifying the DataFrame as requested]
# 4-1.  Converting string into datetime64(without timezone info) which is reserved in EQ_DATA
df["datetime"] = pd.to_datetime(df["time"], format="mixed", errors="coerce",utc=True).dt.tz_localize(None).dt.round("s")
#
# 4-2.  Exporting the new format "datetime" from raw data frame into output data frame.
df_output["datetime"] = df["datetime"]
#
# 4-3.  Calculating the time before earthquake occurence 4hour and exporting it into output data frame.
df_output["4h_before"] = df_output["datetime"] - pd.Timedelta(hours=4)
#
# 4-4.  Exporting the latitude and longtitued from raw data frame into output data frame.
df_output["latitude"] = df["latitude"]
df_output["longitude"] = df["longitude"]
#
# 4-5.  Converting the longitude into 0-360 degree and exporting it into output data frame.
df_output["longitude_360"] = df["longitude"].where(df["longitude"] >= 0, df["longitude"] + 360)
#
# 4-6.  Exporting the depth and magnitude from raw data frame into output data frame.
df_output["depth"] = df["depth"]
df_output["mag"] = df["mag"]
#
# 4-7.  Sorting the output data frame by "datetime" column in ascending order.
df_output = df_output.sort_values(by="datetime", ascending=True).reset_index(drop=True)
#
# 4-8.  Creating "eq_id" as a new column
df_output["eq_id"] = df_output.index
#
# 4-9.  Reordering the columns in output data frame.
df_output = df_output[["eq_id","4h_before","datetime","latitude","longitude","longitude_360","depth","mag"]]
#
# [5.   Exporting the data frame as a csv file]
# 5-1.  Outputing dataframe as a csv file.
df_output.to_csv(OUTPUT_DIR / "eq_m4.8above_depth40kmbelow_2004-2010_preprocessed.csv", index=False)
#
#--------------------------------------------------------------------------------------------------