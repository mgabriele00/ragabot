import csv
import numpy as np
import polars as pl
import os
from typing import List
# Importa StrategyCondition per il type hinting (se non già fatto)
# Potrebbe essere necessario aggiustare il percorso relativo o assoluto
# a seconda della struttura del tuo progetto
from models.strategy_condition import StrategyCondition # Assicurati che questo import funzioni

FOLDER = "dati_forex/EURUSD"  # Nome della cartella contenente i file CSV

def load_forex_data_dohlc(year) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    script_dir = os.path.dirname(__file__)
    # Costruisci il percorso completo alla cartella FOLDER
    # NOTA: Se FOLDER è un percorso assoluto, os.path.join potrebbe non fare quello che ti aspetti.
    # Se FOLDER è sempre lo stesso percorso assoluto, puoi usarlo direttamente.
    # Se FOLDER è relativo allo script_dir, allora os.path.join è corretto.
    # Dato che FOLDER inizia con '/', è un percorso assoluto.
    folder_path = FOLDER # Usa direttamente il percorso assoluto
    try:
        # Lista i file nella cartella specificata
        files_in_folder = os.listdir(folder_path)
        # Filtra i file CSV per l'anno specificato
        files = sorted([f for f in files_in_folder if f.endswith('.csv') and str(year) in f])
    except FileNotFoundError:
        print(f"ERRORE: La cartella dei dati '{folder_path}' non è stata trovata.")
        # Restituisci tuple di array vuoti invece di DataFrame vuoto per coerenza
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if not files:
        print(f"Attenzione: Nessun file CSV trovato per l'anno {year} nella cartella {folder_path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    print("Numero di file trovati: ", len(files))

    dfs = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pl.read_csv(
                full_path, has_header=False,
                new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close']
            ).with_columns(
                pl.concat_str([pl.col("Date"), pl.lit(" "), pl.col("Time")])
                .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M", strict=False) # Usa strict=False per più tolleranza
                .alias("Datetime")
            ).select(["Datetime", "Open", "High", "Low", "Close"])
            dfs.append(df)
        except Exception as e:
            print(f"Errore durante la lettura o elaborazione del file {full_path}: {e}")
            continue # Salta questo file e continua con il prossimo

    if not dfs:
         print(f"Nessun DataFrame caricato con successo per l'anno {year}.")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Concatena i DataFrame
    combined_df = pl.concat(dfs).sort("Datetime")

    # Converte in numpy array
    # Assicurati che le colonne esistano prima di accedervi
    if combined_df.height > 0:
        date = combined_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()
        open_ = combined_df["Open"].to_numpy()
        high = combined_df["High"].to_numpy()
        low = combined_df["Low"].to_numpy()
        close = combined_df["Close"].to_numpy()
        return date, open_, high, low, close
    else:
        print("DataFrame combinato è vuoto dopo l'elaborazione.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


# Modifica save_results_to_csv per accettare anche strategy_conditions
def save_results_to_csv(results: np.ndarray, strategy_conditions: list[StrategyCondition], filename: str):
    """
    Salva i risultati della simulazione e le condizioni della strategia in un file CSV.

    Args:
        results (np.ndarray): Array 2D contenente i risultati (FinalEquity, FinalPnL, MaxDrawdown, WinRate).
        strategy_conditions (list[StrategyCondition]): Lista delle condizioni di strategia usate.
        filename (str): Nome del file CSV in cui salvare i risultati.
    """
    print(f"Saving results to {filename}...")
    # Assicurati che il numero di risultati corrisponda al numero di condizioni
    if len(results) != len(strategy_conditions):
        print(f"Error: Mismatch between number of results ({len(results)}) and number of conditions ({len(strategy_conditions)}). Cannot save CSV.")
        return
    # Verifica che results abbia 4 colonne se ci aspettiamo il Win Rate
    if results.shape[1] != 4:
        print(f"Error: Results array has {results.shape[1]} columns, expected 4 (Equity, PnL, MD, WinRate). Cannot save CSV.")
        return


    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Scrivi l'intestazione: parametri della condizione + metriche di performance
        header = [
            'ConditionIndex',
            'rsi_entry', 'rsi_exit', 'exposure', 'atr_factor', 'bb_std', 'atr_window', 'bb_width_threshold',
            'FinalEquity', 'FinalPnL', 'MaxDrawdown', 'WinRate' # Metriche
        ]
        writer.writerow(header)

        # Scrivi i dati
        for i in range(len(results)):
            condition = strategy_conditions[i]
            # Scrivi l'indice, i parametri della condizione e i risultati della simulazione
            row_data = [
                i,
                condition.rsi_entry, condition.rsi_exit, condition.exposure,
                condition.atr_factor, condition.bb_std, condition.atr_window, condition.bb_width_threshold,
                results[i, 0], results[i, 1], results[i, 2], results[i, 3] # Equity, PnL, MD, WR
            ]
            writer.writerow(row_data)
    print("Results saved.")
    
    
    



def build_polars_table_for_year(
    strategy_conditions: List[StrategyCondition],
    final_equities,
    drawdowns,
    year: int
) -> pl.DataFrame:
    return pl.DataFrame({
        "rsi_entry": [float(c.rsi_entry) for c in strategy_conditions],
        "rsi_exit": [float(c.rsi_exit) for c in strategy_conditions],
        "exposure": [float(c.exposure) for c in strategy_conditions],
        "atr_factor": [float(c.atr_factor) for c in strategy_conditions],
        "bb_std": [float(c.bb_std) for c in strategy_conditions],
        "atr_window": [int(c.atr_window) for c in strategy_conditions],
        "bb_width_threshold": [float(c.bb_width_threshold) for c in strategy_conditions],
        f"equity_{year}": final_equities,
        f"drawdown_{year}": drawdowns
    })

def combine_all_years_by_parameters(
    years: List[int],
    main_fn
) -> pl.DataFrame:
    """
    Esegue il main per ogni anno e unisce i risultati su riga in base alla combinazione di parametri.
    :param years: lista di anni (es. [2013, 2014])
    :param main_fn: funzione main(year) → (final_equities, _, drawdowns)
    :return: DataFrame Polars unificato
    """
    df_merged = None
    strategy_conditions = None

    for year in years:
        final_equities, _, drawdowns = main_fn(year)
        if strategy_conditions is None:
            from models.strategy_condition import generate_conditions_to_test
            strategy_conditions = generate_conditions_to_test()

        df_year = build_polars_table_for_year(strategy_conditions, final_equities, drawdowns, year)

        if df_merged is None:
            df_merged = df_year
        else:
            df_merged = df_merged.join(df_year, on=[
                "rsi_entry", "rsi_exit", "exposure",
                "atr_factor", "bb_std", "atr_window", "bb_width_threshold"
            ])
    return df_merged
