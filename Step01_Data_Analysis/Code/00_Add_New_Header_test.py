import os
import pandas as pd

input_directory = r"C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step01_Data_Analysis\Data\For_Testing"
output_directory = r"C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step01_Data_Analysis\Output\For_Testing"
os.makedirs(output_directory, exist_ok=True)

processed_files_log_file = os.path.join(output_directory, "processed_files_log.csv")

# --- ヘッダーは1回だけ作る ---
header1 = ['year', 'month', 'day', 'hour', 'min', 'sec', 'msec', 'lat', 'lon', 'mlat', 'mlon']
sequence = [i * 19.53125 for i in range(1, 1025)]
hz_headers = [f"{v}Hz" for v in sequence]
new_header = header1 + hz_headers

# --- 処理済みファイル集合（必要なら） ---
processed_files = set()
if os.path.exists(processed_files_log_file):
    df_log = pd.read_csv(processed_files_log_file)
    if "Processed Files" in df_log.columns:
        processed_files = set(df_log["Processed Files"].dropna().astype(str))

new_processed_files = []
file_count = 0

# --- scandirで列挙 ---
for entry in os.scandir(input_directory):
    if not entry.is_file() or not entry.name.endswith(".csv"):
        continue
    file_name = entry.name

    # 既処理スキップ（使うなら有効化）
    if file_name in processed_files:
        continue

    try:
        input_file = entry.path
        output_file = os.path.join(output_directory, file_name)

        # 読み込み（CエンジンのままでOK）
        df = pd.read_csv(input_file, skiprows=1, header=None)
        df.columns = new_header  # ★ コピーを作らない

        df.to_csv(output_file, index=False)

        new_processed_files.append(file_name)
        file_count += 1
        if file_count % 100 == 0:
            print(f"{file_count} ファイルが処理されました。")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# --- ログは追記が軽い（concatのための再読み込み不要） ---
if new_processed_files:
    df_new = pd.DataFrame({"Processed Files": new_processed_files})
    write_header = not os.path.exists(processed_files_log_file)
    df_new.to_csv(processed_files_log_file, mode="a", header=write_header, index=False)

print(f"処理済みファイル名のログが {processed_files_log_file} に保存されました。")
