import pandas as pd
import matplotlib.pyplot as plt

def cumulative_comparison_plots(
    original_eq_file, declustring_eq_file,
    save_path=None    
):

# データの読み取り
df_org = pd.read_csv(original_eq_file)
df_declus = pd.csv(declustring_eq_file)

# 時間をdatetime形に変形
df_org['datetime'] = pd.to_datetime(df_org['year', 'month', 'day', 'hour', 'minute', 'second'])
df_declus['datetime'] = pd.to_datetime(df_declus['year', 'month', 'day', 'hour', 'minute', 'second'])

# 

if __name__ == "__main__":