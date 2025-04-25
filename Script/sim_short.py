import os
import polars as pl
import numpy as np
import talib
import itertools
import pandas as pd
import glob # Importa il modulo glob
import re # Importa il modulo re per le espressioni regolari

INITIAL_CASH = 1000
LEVERAGE = 30
SAVE_EVERY = 100 # Modificato a 100
FOLDER = '../dati_forex/EURUSD/'
RESULTS_FOLDER = 'orders/sim_short' # Definisci la cartella dei risultati principali

PARAM_GRID = {
    "rsi_entry": list(range(30, 46)),
    "rsi_exit": list(range(55, 71)),
    "bb_std": [1.5, 1.75, 2.0],
    "exposure": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "atr_window": [14, 20],
    "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
}

YEARS_INPUT = [2015]

def load_forex_data(year):
    script_dir = os.path.dirname(__file__)
    # Costruisci il percorso completo alla cartella FOLDER
    folder_path = os.path.join(script_dir, FOLDER)
    try:
        # Lista i file nella cartella specificata
        files_in_folder = os.listdir(folder_path)
        # Filtra i file CSV per l'anno specificato
        files = sorted([f for f in files_in_folder if f.endswith('.csv') and str(year) in f])
    except FileNotFoundError:
        print(f"ERRORE: La cartella dei dati '{folder_path}' non è stata trovata.")
        return pl.DataFrame() # Restituisce DataFrame vuoto

    if not files:
        print(f"Attenzione: Nessun file CSV trovato per l'anno {year} nella cartella {folder_path}")
        return pl.DataFrame()

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
            # Puoi decidere se continuare con gli altri file o fermarti
            continue # Salta questo file e continua con il prossimo

    if not dfs:
         print(f"Nessun DataFrame caricato con successo per l'anno {year}.")
         return pl.DataFrame()

    return pl.concat(dfs).sort("Datetime")

def generate_combinations(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def calculate_indicators(df):
    close = df["Close"].to_numpy()
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    rsi = talib.RSI(close, 14)
    bullish, bearish = np.zeros(len(close), bool), np.zeros(len(close), bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        result = getattr(talib, pattern)(open_, high, low, close)
        bullish |= result > 0
        bearish |= result < 0
    return rsi, bullish, bearish

def generate_signals(close, rsi, bullish, bearish, params):
    upper, _, lower = talib.BBANDS(close, 14, params['bb_std'], params['bb_std'])

    entries_long = (rsi < params['rsi_entry']) & (close < lower) & bullish
    exits_long = (rsi > params['rsi_exit']) & (close > upper) & bearish

    entries_short = (rsi > params['rsi_exit']) & (close > upper) & bearish
    exits_short = (rsi < params['rsi_entry']) & (close < lower) & bullish

    return entries_long, exits_long, entries_short, exits_short

def backtest(df, indicators, params, sim_id, year):
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    datetime = df["Datetime"].to_numpy()

    rsi, bullish, bearish = indicators
    entries_long, exits_long, entries_short, exits_short = generate_signals(close, rsi, bullish, bearish, params)
    atr = talib.ATR(high, low, close, params['atr_window'])

    cash, position, orders = INITIAL_CASH, None, []

    for i in range(len(close)):
        price = close[i]

        if not position:
            if entries_long[i]:
                direction = 'LONG'
                sl = price - atr[i] * params['atr_factor']
                tp = price + atr[i] * params['atr_factor'] * 2
            elif entries_short[i]:
                direction = 'SHORT'
                sl = price + atr[i] * params['atr_factor']
                tp = price - atr[i] * params['atr_factor'] * 2
            else:
                continue

            size_eur = cash * params['exposure']
            size = (size_eur * LEVERAGE) / price
            position = {'price': price, 'size': size, 'sl': sl, 'tp': tp, 'time': datetime[i], 'direction': direction}

        elif position:
            exit_reason = None
            if position['direction'] == 'LONG':
                if price <= position['sl']: exit_reason = 'Stop Loss'
                elif price >= position['tp']: exit_reason = 'Take Profit'
                elif exits_long[i]: exit_reason = 'Signal Exit'
            elif position['direction'] == 'SHORT':
                if price >= position['sl']: exit_reason = 'Stop Loss'
                elif price <= position['tp']: exit_reason = 'Take Profit'
                elif exits_short[i]: exit_reason = 'Signal Exit'

            if exit_reason:
                pnl = (price - position['price']) * position['size'] if position['direction'] == 'LONG' else (position['price'] - price) * position['size']
                cash += pnl
                orders.append({
                    "Simulation": sim_id, "Entry Time": position['time'], "Exit Time": datetime[i],
                    "Entry Price": position['price'], "Exit Price": price, "Size": position['size'],
                    "PnL": pnl, "Cash": cash, "Year": year, "Reason": exit_reason, **params, "Direction": position['direction']
                })
                position = None

    return cash, orders

def save_results(orders, year, part_id):
    """Salva un batch di ordini in formato Parquet parziale."""
    if not orders:
        return  # Non salvare file vuoti

    script_dir = os.path.dirname(__file__)
    folder = os.path.join(script_dir, RESULTS_FOLDER, str(year))
    os.makedirs(folder, exist_ok=True)

    filename = f'orders_{year}_part_{part_id}.parquet'
    filepath = os.path.join(folder, filename)

    try:
        pl.DataFrame(orders).write_parquet(filepath, compression='zstd')
        print(f"💾 Parquet parziale salvato: {filepath}")
    except Exception as e:
        print(f"Errore durante il salvataggio del file {filepath}: {e}")


def merge_results(base_folder):
    """Legge tutti i file Parquet parziali, li unisce e salva il risultato finale."""
    script_dir = os.path.dirname(__file__)
    search_path = os.path.join(script_dir, base_folder, '**', 'orders_*_part_*.parquet')
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print("Nessun file Parquet parziale trovato da unire.")
        return

    print(f"Trovati {len(all_files)} file Parquet parziali da unire.")

    try:
        lazy_dfs = [pl.scan_parquet(f) for f in all_files]
        merged_df = pl.concat(lazy_dfs).collect()

        final_filename = os.path.join(script_dir, base_folder, 'merged_all_orders.parquet')
        merged_df.write_parquet(final_filename, compression='zstd')
        print(f"✅ File Parquet finale salvato in: {final_filename}")

        # Opzionale: Rimuovere file parziali dopo l'unione
        # for f in all_files:
        #     os.remove(f)

    except Exception as e:
        print(f"Errore durante l'unione dei file Parquet: {e}")


def get_last_processed_info(year, base_results_path):
    """Trova l'ultimo part_id salvato per un dato anno (Parquet)."""
    script_dir = os.path.dirname(__file__)
    year_folder = os.path.join(script_dir, base_results_path, str(year))
    last_part_id = 0

    if os.path.exists(year_folder):
        try:
            pattern = re.compile(rf'orders_{year}_part_(\d+)\.parquet')
            max_part = 0
            for filename in os.listdir(year_folder):
                match = pattern.match(filename)
                if match:
                    part_num = int(match.group(1))
                    if part_num > max_part:
                        max_part = part_num
            last_part_id = max_part
        except Exception as e:
            print(f"Errore durante scansione {year_folder}: {e}")

    return last_part_id


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    results_base_path = os.path.join(script_dir, RESULTS_FOLDER)
    os.makedirs(results_base_path, exist_ok=True)

    combinations = generate_combinations(PARAM_GRID)
    total_combinations = len(combinations)
    print(f"Numero totale di combinazioni da testare: {total_combinations}")

    for year in YEARS_INPUT:
        print(f"\n===== Inizio elaborazione anno: {year} =====")

        # --- Inizio Modifica: Controllo Ripresa ---
        last_part_id = get_last_processed_info(year, RESULTS_FOLDER)
        combinations_to_skip = last_part_id * SAVE_EVERY
        start_part_id = last_part_id + 1

        if combinations_to_skip > 0:
            if combinations_to_skip >= total_combinations:
                print(f"✅ Anno {year} già completato (trovati {last_part_id} file parziali). Skipping.")
                continue # Salta all'anno successivo
            else:
                print(f"▶️ Riprendendo l'anno {year} dalla combinazione {combinations_to_skip + 1} (trovati {last_part_id} file parziali).")
        # --- Fine Modifica: Controllo Ripresa ---

        try:
            df = load_forex_data(year)
            if df.height == 0:
                print(f"Skipping anno {year} a causa di dati mancanti o errore nel caricamento.")
                continue
            indicators = calculate_indicators(df)
        except Exception as e:
            print(f"Errore durante preparazione dati per anno {year}: {e}")
            print(f"Skipping anno {year}.")
            continue

        all_orders_batch = []
        part_id = start_part_id # Inizia dal part_id corretto

        for idx, params in enumerate(combinations, 1):
            # --- Inizio Modifica: Salta Combinazioni ---
            if idx <= combinations_to_skip:
                continue # Salta questa combinazione perché già processata
            # --- Fine Modifica: Salta Combinazioni ---

            if idx % (SAVE_EVERY * 10) == 0 or idx == combinations_to_skip + 1 or idx == total_combinations:
                 print(f"🔄 Anno {year} | Combinazione {idx}/{total_combinations}")

            try:
                final_cash, orders = backtest(df, indicators, params, idx, year)
                if orders:
                    all_orders_batch.extend(orders)
            except Exception as e:
                print(f"Errore nel backtest per Anno {year}, Comb {idx}, Params {params}: {e}")
                continue

            # Salva batch parziali - La logica qui rimane simile ma usa il part_id aggiornato
            # Controlla se il numero di combinazioni *processate in questa esecuzione* è un multiplo di SAVE_EVERY
            # O più semplicemente, controlla l'indice globale `idx`
            if idx % SAVE_EVERY == 0: # Controlla l'indice assoluto
                save_results(all_orders_batch, year, part_id)
                all_orders_batch = []
                part_id += 1

        # Salva eventuali ordini rimanenti alla fine del loop dell'anno
        if all_orders_batch:
            # L'ultimo part_id sarà quello calcolato nel ciclo o start_part_id se nessuna nuova parte è stata salvata
            save_results(all_orders_batch, year, part_id)

    print("\n===== Unione dei risultati finali =====")
    merge_results(RESULTS_FOLDER) # La funzione merge non necessita modifiche

    print("\n🏁 Elaborazione completata.")
