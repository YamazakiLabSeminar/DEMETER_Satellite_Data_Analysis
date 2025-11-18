## 目録
- 理論
- 内容

## 理論
- 00_データ前処理
	- USGSから、**2004年～2010年まで**に起こった**震源の深さ40km以内**、**マグニチュードM4.8以上**の地震カタログを[csvファイル](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/3db324047bcf3e516c1e425ed0787ece02918a8f/Step00_Declustring/Data/eq_m4.8above_depth40kmbelow_200407-201012.csv)としてダウンロードする。
 	- そのcsvファイルの時間列を**年、月、日、時間、分、秒、ミリ秒**に分けて、新しいcsvファイルとして保存する。
- 01_デクラスタリング（クラスタ除去）
	- [Gardner&Knopoff(1974)](https://doi.org/10.1785/BSSA0640051363)のアルゴリズムを利用することより、デクラスタリングを実行する。
    	- ほかのデクラスタリングのアルゴリズム(**Reasenberg (1985)**, **Uhrhammer (1986)**, **Zaliapin et al. (2008)**)
    - ある地震が発生した後、**特定の時間と距離の範囲内**で発生したマグニチュードがより小さい地震は、すべてその地震の余震とみなして除去する。

- 02_地震累積比較図のプロット
[デクラスタリング前](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/3db324047bcf3e516c1e425ed0787ece02918a8f/Step00_Declustring/Output/eq_m4.8above_depth40kmbelow_200407-201012_add_time_row.csv)、[後](Step00_Declustring\Output\all_eq_declustring_30day_30km.csv)のデータを利用して、時系列順にイベント数の変化を知りたいための[累積比較図](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/e64213b3475cf8a221a42f043eafbc7cdbd6a6d7/Step00_Declustring/graphic/cumulative_cimparison_plots_30days_30km.png)を作成する。
	- x軸は時間、y軸は地震イベントの累積数。
	- [コード](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/e64213b3475cf8a221a42f043eafbc7cdbd6a6d7/Step00_Declustring/Code/Cumulative_Comparison.py)

## 内容
地震カタログをマグニチュード降順および時間列昇順にそれぞれ並べる。まず、最大マグニチュードの地震から走査し、その地震が時間列にある位置を探す。時間列において、現在選択された地震（以降、現在地震と呼ぶ）と以降のそれぞれの地震の時間的・空間的にの差を求め、閾値の範囲内であることを確認しながら走査する。閾値の範囲内の地震（現在地震以外）は、全部削除フラグを立て、以後のループ操作でスキップする。
以上の操作を繰り返し、デクラスタリングを行う。
