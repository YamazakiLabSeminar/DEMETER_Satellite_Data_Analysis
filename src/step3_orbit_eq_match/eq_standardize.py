import pandas as pd
import numpy as np
from pathlib import Path

EQ_DATA = Path(r"E:\tables\earthquake_catalog\declustered\eq_m4.8above_depth40kmbelow_2004_2010_declustered.csv")
OUTPUT_DIR = Path(r"E:\tables\earthquake_catalog\standardize")

df = pd.read_csv(EQ_DATA,usecols=["datetime","latitude","longitude","depth","mag"])
df.info()
print(df)

# Converting string into datetime64(without timezone info) which is reserved in EQ_DATA
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True).dt.tz_localize(None)
df.info()

# Converting datetime format
df["datetime"] = df["datetime"].dt.strftime("%Y/%m/%d  %H:%M:%S")

# Converting the new format "datetime" from object into datetime64
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d  %H:%M:%S")

df.info()
print(df)

# Caculating the time before earthquake occurence
df["4h_before"] = df["datetime"] + pd.Timedelta(hours=4)

