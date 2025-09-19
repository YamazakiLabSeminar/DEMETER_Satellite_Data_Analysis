## 目録
- プロセス
- コード

## プロセス
- 00_データ前処理
	- USGSから、**2004年～2010年まで**に起こった**震源の深さ40km以内**、**マグニチュードM4.8以上**の地震カタログをcsvファイルとしてダウンロードする。
 	- そのcsvファイルの時間列を**年、月、日、時間、分、秒、ミリ秒**に分けて、新しいcsvファイルとして保存する。
- 01_デクラスタリング
 	- 新しく作成した地震カタログをデクラスタリングする。

## コード
### 00_データ前処理
- pandasライブラリを使用する。
<details><summary>コードソース</summary>

```py
import pandas as pd

df = pd.read_csv(r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Earthquake_catalog\Original\EarthquakeCatalog_Depth40kmbelow_Magnitude4.8above.csv', encoding='cp932')
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
print(output_df)
output_df.to_csv(r'C:\Users\nzy27\Documents\Github\DEMETER_Satellite_Data_Analysis\Step00_Declustring\Data\Earthquake_catalog\Time_seperate\EarthquakeCatalog_Depth40kmbelow_Magnitude4.8above_TimeSeperate.csv', index=False, encoding='utf-8-sig')
```
</details>
