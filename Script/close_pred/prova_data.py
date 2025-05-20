import pandas as pd
df = pd.read_parquet("../close_pred/exploded_dataset.parquet")
print(df.columns.tolist())