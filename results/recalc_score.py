import polars as pl

# Percorso al tuo CSV salvato
csv_file = "results/scored_strategies_by_params.csv"


#!/usr/bin/env python3
"""
recalc_score_pandas.py

Script per ricalcolare lo score delle strategie a partire da un CSV preesistente
usando Pandas, e stampare a video la riga con score massimo.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    # 1) Determina la cartella dello script e costruisci il path al CSV
    csv_file = "results/scored_strategies_by_params.csv"

    # 2) Carica il CSV in un DataFrame Pandas
    df = pd.read_csv(csv_file)

    # 3) Identifica le colonne equity_YYYY e drawdown_YYYY
    equity_cols = [col for col in df.columns if col.startswith("equity_")]
    years = [int(col.split("_")[1]) for col in equity_cols]
    drawdown_cols = [f"drawdown_{y}" for y in years]

    # 4) Definisci i pesi per lo score
    alpha, beta, gamma = 0.9, 0.05, 0.05

    # 5) Calcola mean_equity, mean_drawdown e mean_squared_equity
    df["mean_equity"] = df[equity_cols].mean(axis=1)
    df["mean_drawdown"] = df[drawdown_cols].mean(axis=1)
    df["mean_squared_equity"] = (df[equity_cols] ** 2).mean(axis=1)

    # 6) Calcola var_equity, std_equity e rel_volatility
    df["var_equity"] = df["mean_squared_equity"] - df["mean_equity"] ** 2
    df["std_equity"] = np.sqrt(df["var_equity"])
    df["rel_volatility"] = df["std_equity"] / df["mean_equity"]

    # 7) Calcola lo score finale
    df["score"] = (alpha * df["mean_equity"]
                   - beta  * df["mean_drawdown"]
                   - gamma * df["rel_volatility"])

    # 8) Estrai la riga con lo score migliore
    idx_best = df["score"].idxmax()
    best = df.loc[idx_best]

    # 9) Stampa il risultato
    print("Strategia migliore (riga completa):")
    print(best.to_frame().T.to_string(index=False))

if __name__ == "__main__":
    main()
