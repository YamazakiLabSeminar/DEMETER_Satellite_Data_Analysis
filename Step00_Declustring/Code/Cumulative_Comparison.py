import pandas as pd
import matplotlib.pyplot as plt

original_eq_file=r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\eq_m4.8above_depth40kmbelow_200407-201012_add_time_row.csv'
declustring_eq_file=r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Output\all_eq_declustring_30day_30km.csv'
save_path=r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\graphic\cumulative_cimparison_plots_30days_30km.png'



# データの読み取り
df_org = pd.read_csv(original_eq_file)
df_declus = pd.read_csv(declustring_eq_file)

# 時間をdatetime形に変形
df_org['datetime'] = pd.to_datetime(df_org[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_declus['datetime'] = pd.to_datetime(df_declus[['year', 'month', 'day', 'hour', 'minute', 'second']])

# 時系列昇順にソートする
df_org = df_org.sort_values('datetime')
df_declus = df_declus.sort_values('datetime')

# 一応確認
print(f'[Info]This is sorted dataframe.Original:{df_org} Declustring:{df_declus}')

# 両dfに累積カウントの列を追加する
df_org['cumulative_count'] = range(1, len(df_org)+1)
df_declus['cumulative_count'] = range(1, len(df_declus)+1)

# グラフ作成
#==================================================================================================================================
## グラフサイズ（幅76mm*高さ55mm）
fig, ax = plt.subplots(layout="constrained")

## プロット
ax.plot(df_org['datetime'], df_org['cumulative_count'], label='Before', ls="-.", color="black")
ax.plot(df_declus['datetime'], df_declus['cumulative_count'], label='After(30km-30days)', color='black')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Number')
ax.legend()
plt.tight_layout()

# 保存
fig.savefig(save_path, dpi=300)


