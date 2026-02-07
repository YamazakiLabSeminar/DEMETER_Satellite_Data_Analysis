# -*- coding: utf-8 -*-
# orbit_start_end.py
###################################################################################################
###################################################################################################
# Python script to caculate starting time and ending time of each orbit.
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
#INPUT_DIR                  :an object which contains the directory of input directory.
#OUTPUT_DIR                 :an object which contains the directory of output directory.
#allfile_names              :a list which contains the names of files which the orbit data is 
#                            preserved.
#length_allfile_names       :an object which contains length of "allfile_names" object.
#data                       :a two dimensional list object which will contain orbit number,
#                            starting and ending time.
#df                         :a data frame which contains input data.
#
#**************************************************************************************************
# Structure of this script:
#--------------------------------------------------------------------------------------------------
#[1.  Importing modules]
#1-1.  Importing pandas, and pathlib modules.
#
#
#[2.  Setting the directory path]
#2-1.  Setting the input and output directory paht.
#2-2.  Show the directory in the terminal.
#
#
#[3.  Extracting the names of csv files]
#3-1.  Getting the names of files in which the orbit data is preserved.
#3-2.  Sorting the order of file names.
#
#
#[4.  Preparing a list object]
#4-1.  Preparing a list which will contain orbit number(file names), starting time and ending time
#      of each file.
#
#
#[5.  Extracing orbit number(file name), starting and ending time.]
#5-1.  Importing orbit data from a csv file.
#5-2.  Extracting starting and ending time.
#5-3.  Converting strting and ending time in string format into integer format.
#5-4.  Inserting orbit number(file name), starting and ending name into the list object whichi is
#      prepared in 4.
#
#
#[6.  Exporting data in a csv format file]
#6-1.  Converting a list object into a dataframe object.
#6-2.  Defining the names of items(columns).
#6-3.  Exporting the dataframe object into a csv file.
#
#
###################################################################################################
#[1.  Importing modules]
#1-1.  Importing pandas, and pathlib modules.
import pandas as pd
from pathlib import Path
#
#
#[2.  Setting the directory path]
#2-1.  Setting the input and output directory paht.
INPUT_DIR = Path(r"E:\interim\step2_normalized")
OUTPUT_DIR = Path(r"E:\tables")
#
#2-2.  Show the directory in the terminal.
print("Directory of input=", {INPUT_DIR})
print("Directory of output", {OUTPUT_DIR})
#
#
#[3.  Extracting the names of csv files]
#3-1.  Getting the names of files in which the orbit data is preserved.
allfile_names = INPUT_DIR.glob("*.csv")
#
#3-2.  Sorting the order of file names as a list
allfile_names = sorted(allfile_names)
#
#
#[4.  Preparing a list object]
#4-1.  Preparing a list which will contain orbit number(file names), starting time and ending time
#      of each file.
length_allfile_names = len(allfile_names)
data = []
#
#
#[5.  Extracing orbit number(file name), starting and ending time.]
for f in range(length_allfile_names):
#   
#5-1.  Importing orbit data from a csv file.
    df = pd.read_csv(allfile_names[f])
#5-2.  Extracting starting and ending time.
    start_t = pd.to_datetime(df["datetime"].iloc[0])
    end_t   = pd.to_datetime(df["datetime"].iloc[-1])
#
#5-3.  Converting starting and ending time in string format into integer format.
    start_t = int(start_t.strftime("%Y%m%d%H%M%S"))
    end_t   = int(end_t.strftime("%Y%m%d%H%M%S"))    
#
#5-4.  Inserting orbit number(file name), starting and ending name into the list object whichi is
#      prepared in 4.
    data.append((allfile_names[f].name, start_t, end_t))
#
#
#[6.  Exporting data in a csv format file]
#6-1.  Converting a list object into a dataframe object.
data = pd.DataFrame(data, columns=["orbit_file", "start_time", "end_time"])
#
#6-2.  Exporting the dataframe object into a csv file.
data.to_csv(r"E:\tables\orbit_start_end.csv", index=False)
#
#
