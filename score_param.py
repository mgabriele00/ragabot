import os
import pickle
import numpy as np
import pandas as pd
import polars as pl
import talib
from glob import glob
from itertools import product
from typing import Dict, List, Tuple, Any

# === Parametri iniziali ===
INITIAL_CASH = 1000
param_ranges = {
    "sl": [0.002, 0.004, 0.006, 0.008, 0.01],
    "tp": [0.01, 0.015, 0.02, 0.025, 0.03],
    "rsi_entry": list(range(30, 46, 5)),
    "rsi_exit": list(range(55, 71, 5)),
    "bb_std": [1.5, 1.75, 2.0, 2.25, 2.5],
    "exposure": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

param_keys = list(param_ranges.keys())
params_list = [dict(zip(param_keys, values)) for values in product(*param_ranges.values())]

def calcola_capital_risk(cash_series):
    min_cash = min(cash_series)
    return max(0, min(1.0, (INITIAL_CASH - min_cash) / INITIAL_CASH))

def format_for_excel(df: pl.DataFrame) -> pl.DataFrame:
    param_cols = {"sl", "tp", "bb_std", "exposure"}
    new_cols = []

    for col in df.columns:
        if df.schema[col] == pl.Float64:
            if col in param_cols:
                new_col = pl.col(col).map_elements(lambda x: f"{x:.4f}".replace('.', ',')).alias(col)
            else:
                new_col = pl.col(col).map_elements(lambda x: f"{x:.2f}".replace('.', ',')).alias(col)
            new_cols.append(new_col)
        else:
            new_cols.append(pl.col(col))
    return df.select(new_cols)

# === Lettura e aggregazione dei risultati annuali ===
years = range(2013, 2024)
yearly_dfs = []

for year in years:
    print(f"📅 Elaborando anno {year}")
    files = sorted(glob(f"orders/partial/{year}/orders_{year}_block_*.pkl"))
    year_results = []
    comb_id = 0

    for file in files:
        with open(file, "rb") as f:
            buffer_orders = pickle.load(f)

        for final_cash, orders in buffer_orders:
            if not orders:
                comb_id += 1
                continue

            # Trova l'ordine con i parametri completi
            param_order = next((o for o in orders if all(k in o for k in ['SL', 'TP', 'RSI Entry', 'RSI Exit', 'BB Std', 'Exposure'])), None)

            if param_order is None:
                print(f"❗ Nessun ordine valido per parametri in combinazione {comb_id} - file: {file}")
                comb_id += 1
                continue

            params = {
                'sl': round(param_order['SL'], 5),
                'tp': round(param_order['TP'], 5),
                'rsi_entry': int(param_order['RSI Entry']),
                'rsi_exit': int(param_order['RSI Exit']),
                'bb_std': round(param_order['BB Std'], 5),
                'exposure': round(param_order['Exposure'], 5),
            }

            cash_series = [order['Cash'] for order in orders]
            capital_risk = calcola_capital_risk(cash_series)

            year_results.append({
                'combinazione_id': comb_id,
                **params,
                f'cap_{year}': final_cash,
                f'capital_risk_{year}': capital_risk,
            })
            comb_id += 1

    df_year = pl.DataFrame(year_results).with_columns(
        pl.col(f'cap_{year}').rank(method='ordinal', descending=True).alias(f'rank_{year}')
    )
    yearly_dfs.append(df_year)

# === Join progressivo su chiavi sicure ===
join_keys = ['combinazione_id', 'sl', 'tp', 'rsi_entry', 'rsi_exit', 'bb_std', 'exposure']
final_df = yearly_dfs[0]

for next_df in yearly_dfs[1:]:
    new_cols = [col for col in next_df.columns if col not in final_df.columns or col in join_keys]
    cleaned_df = next_df.select(new_cols)
    final_df = final_df.join(cleaned_df, on=join_keys, how='inner')

# === Calcolo metriche ===
cap_cols = [f'cap_{y}' for y in years if f'cap_{y}' in final_df.columns]
risk_cols = [f'capital_risk_{y}' for y in years if f'capital_risk_{y}' in final_df.columns]
rank_cols = [f'rank_{y}' for y in years if f'rank_{y}' in final_df.columns]

final_df = final_df.with_columns([
    pl.concat_list(cap_cols).list.mean().alias('cap_media'),
    pl.concat_list(risk_cols).list.mean().alias('risk_media'),
    pl.concat_list(cap_cols).list.var().alias('var_capitale'),
    pl.concat_list(risk_cols).list.var().alias('var_risk'),
    pl.concat_list(rank_cols).list.mean().alias('rank_medio'),
])

# === Normalizzazione e score finale ===
final_df = final_df.with_columns([
    ((pl.col('cap_media') - pl.col('cap_media').mean()) / pl.col('cap_media').std()).alias('cap_norm'),
    ((pl.col('risk_media') - pl.col('risk_media').mean()) / pl.col('risk_media').std()).alias('risk_norm'),
    ((pl.col('var_capitale') - pl.col('var_capitale').mean()) / pl.col('var_capitale').std()).alias('var_cap_norm'),
    ((pl.col('var_risk') - pl.col('var_risk').mean()) / pl.col('var_risk').std()).alias('var_risk_norm'),
    ((pl.col('rank_medio') - pl.col('rank_medio').mean()) / pl.col('rank_medio').std()).alias('rank_norm'),
])

final_df = final_df.with_columns(
    (pl.col('cap_norm') * 2
     - pl.col('risk_norm') * 2
     - pl.col('var_cap_norm')
     - pl.col('var_risk_norm')
     - pl.col('rank_norm') * 2
    ).alias('score')
)

final_df = final_df.sort('score', descending=True)

# === Salvataggio output ===
os.makedirs("results", exist_ok=True)

with open("results/final_consistency.pkl", "wb") as f:
    pickle.dump(final_df, f)

formatted_df = format_for_excel(final_df)
formatted_df.write_csv("results/final_consistency_excel.csv", separator=';')

# === Output top 1 ===
best = final_df.head(1).to_dicts()[0]
print("\n🏆 Miglior combinazione trovata:")
print(f"• SL = {best['sl']}, TP = {best['tp']}, RSI Entry = {best['rsi_entry']}, RSI Exit = {best['rsi_exit']}")
print(f"• BB Std = {best['bb_std']}, Exposure = {best['exposure']}")
print(f"• Score = {best['score']:.2f}")
print(f"• Capitale medio = € {best['cap_media']:.2f}")
print(f"• Capitale minimo = -{best['risk_media']:.2%} sotto i 1000€ iniziali")

print("\n✅ File salvati:")
print("📦 Pickle: results/final_consistency.pkl")
print("📄 CSV Excel IT: results/final_consistency_excel.csv")