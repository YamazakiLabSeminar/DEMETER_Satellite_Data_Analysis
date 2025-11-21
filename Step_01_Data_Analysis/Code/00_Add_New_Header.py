import pandas as pd
import numpy as np
import os
import zipfile
import tempfile

# Path of Input folder and Output folder
# For testing
#INPUT_DIR = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step_01_Data_Analysis\Data\For_Testing'
#OUTPUT_DIR = r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step_01_Data_Analysis\Output\For_Testing'

INPUT_DIR = r'F:/01_EFdata'
OUTPUT_DIR =r'F:/01_EFdata_new_header'

# 圧縮ファイルのパス
COMPRESSED_OUTPUT = os.path.join(OUTPUT_DIR, '01_EFdata_new_header_compressed.zip')

# Path of log file of Processed file
PROCESSED_FILES_LOG_FILE = os.path.join(OUTPUT_DIR, 'processed_files_log.csv')

# 一時dirの作成でメモリ消費量を抑える
temp_dir = tempfile.mkdtemp()

# log file読み取る
if os.path.exists(PROCESSED_FILES_LOG_FILE):
    processed_files_df = pd.read_csv(PROCESSED_FILES_LOG_FILE)
    processed_files = set(processed_files_df['Processed_Files'])
else:
    processed_files = set()

# 処理済ファイルのログ用リストを作成
new_preocessed_files_list = []
file_count = 0

# Read the file in input folder as a list
file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

# Zipファイルを開く(追記モード)
with zipfile.ZipFile(COMPRESSED_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zipf:

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

            # new dataframe
            df_with_new_header = pd.DataFrame(input_file_df.values, columns=new_header)

            # 一時ファイルに保存してからZIPに追加
            temp_csv_path = os.path.join(temp_dir, file_name)
            df_with_new_header.to_csv(temp_csv_path, index=False)

            # ZIPファイルに追加
            zipf.write(temp_csv_path, file_name)

            # 一時ファイル削除
            os.remove(temp_csv_path)

            # 処理終わったら処理済ファイルのログ用リストに追加
            new_preocessed_files_list.append(file_name)
            file_count += 1

            # 100個になったら一応報告
            if file_count % 100 == 0:
                print(f"[Info] Processing:{file_count}/{len(file_list)}")

        except Exception as e:
            print(f'Error processing {file_name}: {str(e)}')

# 一時ディレクトリを削除
try:
    os.rmdir(temp_dir)
except:
    pass

# 新しい処理済みファイルのログを更新
if new_preocessed_files_list:
    new_processed_files_df = pd.DataFrame(new_preocessed_files_list, columns=['Processed_Files'])
    if os.path.exists(PROCESSED_FILES_LOG_FILE):
        old_processed_files_df = pd.read_csv(PROCESSED_FILES_LOG_FILE)
        updated_processed_files_df = pd.concat([old_processed_files_df, new_processed_files_df], ignore_index=True)
    else:
        updated_processed_files_df = new_processed_files_df
    updated_processed_files_df.to_csv(PROCESSED_FILES_LOG_FILE, index=False)

print(f'[Info] 処理済みファイル名のログが {PROCESSED_FILES_LOG_FILE} に保存されました。')
print(f'[Info] すべて処理済データが{COMPRESSED_OUTPUT}に圧縮保存されました')
print(f'[Info] 合計{file_count}個のファイルを処理しました')