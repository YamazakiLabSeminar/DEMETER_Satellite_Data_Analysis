import pandas as pd

DATA = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\eq_m4.8above_depth40kmbelow_200407-201012.csv'
OUTPUT_FILE = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\eq_m4.8above_depth40kmbelow_200407-201012_add_time_row.csv'
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
                          'magnitude':df['mag'],
                          'depth':df['depth']
                          })

output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')