import os
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.abspath(os.path.join(script_dir, '../../dati_forex/EURUSD/'))
output_dir = os.path.join(script_dir, 'data')
os.makedirs(output_dir, exist_ok=True)
files = [f for f in os.listdir(input_dir)
         if f.startswith('DAT_MT_EURUSD_M1_2013') and f.endswith('.csv')]
files.sort()
dfs = []
for filename in files:
    path = os.path.join(input_dir, filename)
    df = pd.read_csv(
        path,
        header=None,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Save combined CSV in the newly created 'data' folder
output_file = os.path.join(output_dir, 'EURUSD_M1_2013_2024.csv')
combined.to_csv(output_file, index=False)

print(f"CSV consolidato salvato in: {output_file}")
