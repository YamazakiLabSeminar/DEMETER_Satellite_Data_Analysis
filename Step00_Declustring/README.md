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
import pandas

df = pd.read_csv(r'C:\卒業研究\Exercise\練習問題3_データ.csv', encoding='cp932')
time_list = pd.to_datetime(df['time'])
output_df = pd.DataFrame({'年':time_list.dt.year,
                          '月':time_list.dt.month,
                          '日':time_list.dt.day,
                          '時間':time_list.dt.hour,
                          '分':time_list.dt.minute,
                          '秒':time_list.dt.second,
                          'ミリ秒':time_list.dt.microsecond,
                          '緯度':df['latitude'],
                          '経度':df['longitude'],
                          'マグニチュード':df['mag']
                          })
print(output_df)
output_df.to_csv(r'C:\卒業研究\Exercise\練習問題3_解答.csv', index=False, encoding='cp932')
```
	</details>
