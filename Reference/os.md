## OSモジュール
- OsモジュールはPythonでオペレーティングシステム（OS）に関連する機能を利用するための強力なツールです。**ファイル**や**ディレクトリの操作**、**環境変数の取得**、**プロセス管理**など、様々な機能を提供します。[Qiita](https://qiita.com/automation2025/items/19568653742a64b28f65)

## インストール
- Terminalで操作する。
<details><summary>コードソース</summary>

```python
import os
--- osモジュールの関数を使用する ---
current_dir = os.getcwd()
print(f'現在のディレクトリ: {current_dir}')
```
</details>

## ディレクトリの操作
- 新しいディレクトリを作成し、そのディレクトリに移動し、最後に元のディレクトリに戻る方法。
<details><summary>ディレクトリの操作</summary>

```python
import os
--- 現在のディレクトリを取得 ---
original_dir = os.getcwd()
print(f"元のディレクトリ: {original_dir}")

--- 新しいディレクトリを作成 ---
os.mkdir("新しいフォルダ")

--- 新しいディレクトリに移動 ---
os.chdir("新しいフォルダ")
print(f"新しいディレクトリ: {os.getcwd()}")

--- 元のディレクトリに戻る ---
os.chdir(original_dir)
print(f"元のディレクトリに戻りました: {os.getcwd()}")
```

</details>

## ファイルの操作
- ファイルの名前の変更、削除、存在確認などが可能である。以下のコードで、それぞれの一連操作を示す。
<details><summary>ファイルの操作</summary>
  
```python
import os

# ファイルを作成
with open("test.txt", "w") as f:
    f.write("これはテストファイルです。")

# ファイルの存在を確認
if os.path.exists("test.txt"):
    print("ファイルが作成されました。")

# ファイルの名前を変更
os.rename("test.txt", "renamed_test.txt")
print("ファイル名を変更しました。")

# ファイルを削除
os.remove("renamed_test.txt")
print("ファイルを削除しました。")

# 削除されたことを確認
if not os.path.exists("renamed_test.txt"):
    print("ファイルが削除されたことを確認しました。")

```

</details>
