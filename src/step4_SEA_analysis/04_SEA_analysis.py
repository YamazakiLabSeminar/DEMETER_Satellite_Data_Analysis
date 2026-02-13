import pandas as pd
from pathlib import Path
import pprint

# [ Path Settings ]
#==================================================================================================
DATA_DIR = Path(r"E:\interim\orbit_data_for_sea_analysis_+-200s")
OUTPUT_DIR = Path(r"E:\figuies\sea_analysis_results")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"File not found: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#------------------------------------------------------------------------------------------------

# [ SEA Analysis Settings ]
#==================================================================================================
TIME_WINDOW = 200  # seconds
TIME_RESOLUTION = 2  # seconds

# 出力用dataframe
time_columns = list(range(-TIME_WINDOW, TIME_WINDOW+TIME_RESOLUTION, TIME_RESOLUTION))
columns = ["event_id", "orbit_num"] + [f"t_{t}" for t in time_columns]

sea_df = pd.DataFrame(columns=columns)
# sea_df.info()
# print(sea_df)
#------------------------------------------------------------------------------------------------

# [ SEA Analysis ]
#==============================================================================================
# 1. データディレクトリ内の全CSVファイルを取得
file_names = sorted(DATA_DIR.glob("*.csv"))
pprint.pprint(file_names)
# 2. 各ファイルに対してSEA分析を実行
rows = []
for file_path in file_names:
    stem = file_path.stem  # 例: DMT_N1_1132_34280.1_eq7018
    if "_eq" not in stem:
        print(f"[Skip] unexpected file name format: {file_path.name}")
        continue

    orbit_num, eq_part = stem.rsplit("_eq", 1)
    if not eq_part.isdigit():
        print(f"[Skip] eq_id is not numeric: {file_path.name}")
        continue

    eq_id = int(eq_part)
    rows.append({
        "event_id": eq_id,
        "orbit_num": orbit_num,
    })

sea_df = pd.DataFrame(rows, columns=columns)
print(sea_df[["event_id", "orbit_num"]].head())
