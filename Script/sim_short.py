import os
import polars as pl
import numpy as np
import talib
import itertools
import pandas as pd
import glob # Importa il modulo glob
import re # Importa il modulo re per le espressioni regolari
import numba as nb
from numba import prange

INITIAL_CASH = 1000
LEVERAGE = 30
SAVE_EVERY = 100 # Modificato a 100
FOLDER = '../dati_forex/EURUSD/'
RESULTS_FOLDER = 'orders/sim_short' # Definisci la cartella dei risultati principali

PARAM_GRID = {
    "rsi_entry": [30],
    "rsi_exit": [55],
    "bb_std": [1.5, ],
    "exposure": [0.3 ],
    "atr_window": [14 ],
    "atr_factor": [1.0 ]
}

YEARS_INPUT = [2024]

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

@nb.njit(fastmath=True)
def backtest_core_parallel(close, high, low, datetime_idx,
                           rsi, bullish, bearish, atr,
                           rsi_entry, rsi_exit, bb_std,
                           exposure, atr_factor, leverage,
                           initial_cash):
    n = close.shape[0]
    # prealloc BBANDS
    sma = np.zeros(n, np.float64)
    upper = np.zeros(n, np.float64)
    lower = np.zeros(n, np.float64)
    # Bollinger Bands
    for i in prange(14, n):       # qui ok parallelizzato
        # calcolo BB
        s = 0.0
        for j in range(i-14, i):
            s += close[j]
        m = s / 14.0
        var = 0.0
        for j in range(i-14, i):
            d = close[j] - m
            var += d*d
        std = np.sqrt(var/14.0)
        sma[i] = m
        upper[i] = m + bb_std*std
        lower[i] = m - bb_std*std

    cash = initial_cash
    has_pos = False
    price0 = size = sl = tp = 0.0
    time0 = 0
    dir0 = 0  # 1=LONG,2=SHORT

    # worst‑case orders
    maxo = n
    et = np.empty(maxo, np.int64)
    xt = np.empty(maxo, np.int64)
    ep = np.empty(maxo, np.float64)
    xp = np.empty(maxo, np.float64)
    sz = np.empty(maxo, np.float64)
    pnl = np.empty(maxo, np.float64)
    cash_a = np.empty(maxo, np.float64)
    dr = np.empty(maxo, np.int64)
    rr = np.empty(maxo, np.int64)
    oc = 0

    for i in range(n):            # SIMULAZIONE sequenziale!
        price = close[i]
        if not has_pos:
            if (rsi[i] < rsi_entry) and (price < lower[i]) and bullish[i]:
                dir0 = 1
            elif (rsi[i] > rsi_exit) and (price > upper[i]) and bearish[i]:
                dir0 = 2
            else:
                continue
            # entry
            has_pos = True
            price0 = price
            sl = price - (atr[i]*atr_factor if dir0==1 else -atr[i]*atr_factor)
            tp = price + (atr[i]*atr_factor*2 if dir0==1 else -atr[i]*atr_factor*2)
            size = (cash * exposure * leverage) / price
            time0 = i

        else:
            exit_r = 0
            if dir0 == 1:
                if price <= sl:          exit_r = 1
                elif price >= tp:        exit_r = 2
                elif (rsi[i] > rsi_exit) and (price > upper[i]) and bearish[i]: exit_r = 3
            else:
                if price >= sl:          exit_r = 1
                elif price <= tp:        exit_r = 2
                elif (rsi[i] < rsi_entry) and (price < lower[i]) and bullish[i]: exit_r = 3

            if exit_r>0:
                # calc pnl
                if dir0 == 1:
                    x = (price - price0)*size
                else:
                    x = (price0 - price)*size
                cash += x
                # record
                et[oc] = time0
                xt[oc] = i
                ep[oc] = price0
                xp[oc] = price
                sz[oc] = size
                pnl[oc] = x
                cash_a[oc] = cash
                dr[oc] = dir0
                rr[oc] = exit_r
                oc += 1
                has_pos = False

    return cash, oc, et[:oc], xt[:oc], ep[:oc], xp[:oc], sz[:oc], pnl[:oc], cash_a[:oc], dr[:oc], rr[:oc]


def backtest(df, indicators, params, sim_id, year):
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    dt = df["Datetime"].to_numpy()
    rsi, bullish, bearish = indicators
    # seleziona ATR in base al parametro
    atr = talib.ATR(high, low, close, params["atr_window"])

    # esegue la simulazione per i parametri correnti
    dt_idx = np.arange(len(close), dtype=np.int64)
    cash_end, oc, et, xt, ep, xp, sz, pnl, cash_a, dr, rr = backtest_core_parallel(
        close, high, low, dt_idx,
        rsi, bullish, bearish, atr,
        params["rsi_entry"], params["rsi_exit"], params["bb_std"],
        params["exposure"], params["atr_factor"],
        LEVERAGE, INITIAL_CASH
    )

    # costruisce lista di ordini
    orders = []
    reason_map = {1: "Stop Loss", 2: "Take Profit", 3: "Signal Exit"}
    for i in range(oc):
        orders.append({
            "Simulation": sim_id,
            "Entry Time": dt[et[i]],
            "Exit Time": dt[xt[i]],
            "Entry Price": ep[i],
            "Exit Price": xp[i],
            "Size": sz[i],
            "PnL": pnl[i],
            "Cash": cash_a[i],
            "Year": year,
            "Reason": reason_map[int(rr[i])],
            "Direction": "LONG" if dr[i] == 1 else "SHORT",
            **params
        })
    return cash_end, orders

def save_results(orders, year, part_id):
    """Salva un batch di ordini in formato CSV parziale."""
    if not orders:
        return  # Non salvare file vuoti

    script_dir = os.path.dirname(__file__)
    folder = os.path.join(script_dir, RESULTS_FOLDER, str(year))
    os.makedirs(folder, exist_ok=True)

    filename = f'orders_{year}_part_{part_id}.csv' # Modifica estensione
    filepath = os.path.join(folder, filename)

    try:
        df = pd.DataFrame(orders)
        df.to_csv(filepath, index=False) # Modifica: salva in CSV
        print(f"💾 CSV parziale salvato: {filepath}")

    except Exception as e:
        print(f"Errore durante il salvataggio del file {filepath}: {e}")




def merge_results(base_folder):
    """Legge tutti i file CSV parziali, li unisce e salva il risultato finale."""
    script_dir = os.path.dirname(__file__)
    search_path = os.path.join(script_dir, base_folder, '**', 'orders_*_part_*.csv') # Modifica estensione
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print("Nessun file CSV parziale trovato da unire.")
        return

    print(f"Trovati {len(all_files)} file CSV parziali da unire.")

    try:
        dfs = [pd.read_csv(f) for f in all_files] # Modifica: leggi CSV
        merged_df = pd.concat(dfs, ignore_index=True)

        final_filename = os.path.join(script_dir, base_folder, 'merged_all_orders.csv') # Modifica estensione
        merged_df.to_csv(final_filename, index=False) # Modifica: salva in CSV
        print(f"✅ File CSV finale salvato in: {final_filename}")

    except Exception as e:
        print(f"Errore durante l'unione dei file CSV: {e}")



def get_last_processed_info(year, base_results_path):
    """Trova l'ultimo part_id salvato per un dato anno (CSV)."""
    script_dir = os.path.dirname(__file__)
    year_folder = os.path.join(script_dir, base_results_path, str(year))
    last_part_id = 0

    if os.path.exists(year_folder):
        try:
            pattern = re.compile(rf'orders_{year}_part_(\d+)\.csv') # Modifica estensione
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
                print(f"  -> Anno {year} | Comb {idx} | Params: {params} | Final Cash: {final_cash:.2f}") # <-- Aggiungi questa riga
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
