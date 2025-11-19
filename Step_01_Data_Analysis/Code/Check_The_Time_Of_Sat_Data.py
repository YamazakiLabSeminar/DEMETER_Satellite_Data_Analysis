import pandas as pd
import numpy as np
import os

SSD_DIRECTORY = r'F:/'
Sat_Data_Dic = r'F:/01_EFdata'

current_dir = os.chdir(SSD_DIRECTORY)

print(f'[Info] 現在のディレクトリ:{os.getcwd()}')

for time_dir in current_dir:

    if os.path.exists('F:/02_Selected_by_Time'):
        continue
    else:
        time_dir = os.mkdir(r'F:/02_Selected_by_Time')

current_dir = os.chdir(time_dir)
print(f'[Info] 現在のdir:{os.getcwd()}')