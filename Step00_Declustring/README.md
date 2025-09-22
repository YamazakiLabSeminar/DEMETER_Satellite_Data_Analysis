## 目録
- 理論
- 内容

## 理論
- 00_データ前処理
	- USGSから、**2004年～2010年まで**に起こった**震源の深さ40km以内**、**マグニチュードM4.8以上**の地震カタログをcsvファイルとしてダウンロードする。
 	- そのcsvファイルの時間列を**年、月、日、時間、分、秒、ミリ秒**に分けて、新しいcsvファイルとして保存する。
- 01_デクラスタリング（クラスタ除去）
 	- [Gardner&Knopoff(1974)](https://doi.org/10.1785/BSSA0640051363)のアルゴリズムを利用することより、デクラスタリングを実行する。
    	- ほかのデクラスタリングのアルゴリズム(**Reasenberg (1985)**, **Uhrhammer (1986)**, **Zaliapin et al. (2008)**)

    - ある地震が発生した後、**特定の時間と距離の範囲内**で発生したマグニチュードがより小さい地震は、すべてその地震の余震とみなして除去する。
    - マグニチュードに応じた**時間的ウィンドウ**と**空間的ウィンドウ**を[Gardner&Knopoff(1974)の経験則](https://doi.org/10.1785/BSSA0640051363)に基づいて、計算する。

## 内容
### 環境
- VsCode Python 3.12.1(16-bit)
- ライブラリの使用：
  - pandas2.2.1
  - [os](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/15bf2f1f95f5279023fe3d9d450addb490ac84a1/Reference/os.md)
  - numpy 1.26.4
  - matplotlib 3.8.0
  - Cartopy 0.22.0

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

### 01_デクラスタリング（除去）
- 1\. **00_データ前処理**で新しく作成したcsvファイルを地震発生時刻の新しい順にソートする。
- 2\. **Gardner&Knopoffの経験則**より、ルックアップテーブルを作成し、線形補間関数を作成し定義する。
- 
