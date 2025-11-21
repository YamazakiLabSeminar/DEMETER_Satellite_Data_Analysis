import pandas as pd
import numpy as np
import os

# Path of Input folder and Output folder
INPUT_DIR = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step_01_Data_Analysis\Data\For_Testing'
OUTPUT_DIR = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step_01_Data_Analysis\Output\For_Testing'

# Path of log file of Processed file
PROCESSED_FILES_LOG_FILE = os.path.join(OUTPUT_DIR, 'processed_files_log.csv')

# log file読み取る
if os.path.exists(PROCESSED_FILES_LOG_FILE):
    processed_files_df = pd.read_csv(PROCESSED_FILES_LOG_FILE)
    processed_files = set(processed_files_df['Processed_Files'])
else:
    processed_files = set()

# 処理済ファイルのログ用リストを作成
new_preocessed_list = []
file_count = 0

# Read the file in input folder as a list
file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

# メイン処理
for index, file_name in enumerate(file_list):
    try:
        input_file = os.path.join(INPUT_DIR, file_name)

        # ファイルを読み込み
        input_file_df = pd.read_csv(input_file, skiprows=1, header=None)

        # [Hz]単位のヘッダ作成ための等差数列を作成
        sequence = [i*19.53125 for i in range(1, 1025)]

        # [Hz]単位のヘッダ作成
        Hz_headers = [f"{value}Hz" for value in sequence]
        #確認用
        #print(f"ヘッダー数: {len(Hz_headers)}")
        #print(f"最初のヘッダー: {Hz_headers[0]}")
        #print(f"最後のヘッダー: {Hz_headers[-1]}")
        #print(f"最初の5つ: {Hz_headers[:5]}")
        #print(f"最後の5つ: {Hz_headers[-5:]}")

        header1 = ['year', 'month', 'day', 'hour', 'min', 'sec', 'msec', 'lat', 'lon', 'mlat', 'mlon']
        new_header = header1 + Hz_headers
        
    except Exception as e:
        print(f'Error processing {file_name}: {str(e)}')
