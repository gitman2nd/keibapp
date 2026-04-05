import pandas as pd

# データの読み込み
df = pd.read_parquet("data/ur_bunseki.parquet")

# 1. 最初の5行を表示
print("--- データのプレビュー ---")
print(df.head())

# 2. 列名の一覧とデータ型、欠損値の確認
print("\n--- カラム情報 ---")
print(df.info())

# 3. 統計量の確認（数値データのみ）
print("\n--- 統計要約 ---")
print(df.describe())
