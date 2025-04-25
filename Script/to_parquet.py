import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

folder_path = '/Users/raffaele/Documents/GitHub/ragabot/Script/orders/sim_short/2015'
output_file = '/Users/raffaele/Documents/GitHub/ragabot/Script/orders/sim_short/final/2015_final.parquet'
# 🔥 CREA LA CARTELLA se non esiste
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Lista file
pkl_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])

# Prepariamo il writer Parquet
writer = None

for idx, pkl_file in enumerate(pkl_files):
    print(f"Processando {idx+1}/{len(pkl_files)}: {pkl_file}")

    # carica uno per uno
    df = pd.read_pickle(pkl_file)

    # converte in pyarrow Table
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        # crea il file parquet alla prima iterazione
        writer = pq.ParquetWriter(output_file, table.schema, compression='zstd')

    # scrive il pezzo
    writer.write_table(table)

    # libera memoria
    del df
    del table

# chiude il writer
if writer:
    writer.close()

print("Fatto! 🎯")
