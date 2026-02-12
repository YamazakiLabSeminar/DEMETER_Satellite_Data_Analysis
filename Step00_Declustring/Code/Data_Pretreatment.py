import pandas as pd

DATA = r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv"
OUTPUT_FILE = r"E:\tables\earthquake_catalog\add_time_column\eq_m4.8above_depth40kmbelow_2004-2010_add_time_row_ver2.csv"
df = pd.read_csv(DATA, encoding='utf-8-sig')
time_list = pd.to_datetime(df['time'])
output_df = pd.DataFrame({'year':time_list.dt.year,
                          'month':time_list.dt.month,
                          'day':time_list.dt.day,
                          'hour':time_list.dt.hour,
                          'minute':time_list.dt.minute,
                          'second':time_list.dt.second,
                          'microsecond':time_list.dt.microsecond,
                          'latitude':df['latitude'],
                          'longitude':df['longitude'],
                          'mag':df['mag'],
                          'depth':df['depth']
                          })

output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')