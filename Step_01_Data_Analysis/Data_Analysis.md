## Do What?
1. 地震軌道の抽出 
- [デクラスタリングした本震](https://github.com/YamazakiLabSeminar/DEMETER_Satellite_Data_Analysis/blob/ea46adab4ac50275af3d93d4852a756e9e6ecd12/Step00_Declustring/Output/all_eq_declustring_30day_30km.csv)に対して、地震発生前4時間以内に震央から**半径330km以内**にDEMETER衛星が**700km高度上空**を通った軌道の数を調べる(以下、地震軌道と呼ぶ)。

## How To Do
0. データの前処理
- ヘッダ付き
    - 自分でデータファイルを開いて九九人すると列名が["year", "month", "date", "hour", "min", "sec", "milsec"]で書いてある．pd.to_datetime()で読み取れない．
    
    - 新しいヘッダ付きをする必要がある．["year", "month", "day", "hour", "minute", "second"]

- 空白セルの処理
    - まだ、最悪のは、偶数行の時間列は空白である．ややこしいけど、このままやるとエラーが生じる．そのため、何とかしないと進めない．

    - 