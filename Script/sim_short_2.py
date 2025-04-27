import os
import re
import numpy as np
import talib
import itertools
import pandas as pd
import pyfolio as pf
import time
import csv
import multiprocessing
from functools import partial
import argparse # Aggiunto argparse

INITIAL_CASH = 1000
LEVERAGE = 100
FOLDER = '../dati_forex/EURUSD/'
SUMMARY_RESULTS_FILE = 'pyfolio_summary_results.csv'
PARTIAL_RESULTS_FOLDER = 'pyfolio_partial_results'
SAVE_EVERY = 1000

PARAM_GRID = {
    "rsi_entry": list(range(30, 46)),
    "rsi_exit": list(range(55, 71)),
    "bb_std": [1.5, 1.75, 2.0],
    "exposure": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "atr_window": [14, 20],
    "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
}

# YEARS_INPUT = [2013, 2014, 2015] # Rimosso default hardcoded

# --- Funzioni load_forex_data, generate_combinations, calculate_indicators, generate_signals, backtest ---
# --- (invariate) ---
def load_forex_data(year):
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.abspath(os.path.join(script_dir, FOLDER))
    print(f"DEBUG: Tentativo di accesso alla cartella dati: {folder_path}")
    try:
        files_in_folder = os.listdir(folder_path)
        files = sorted([f for f in files_in_folder if f.endswith('.csv') and str(year) in f])
    except FileNotFoundError:
        print(f"ERRORE: La cartella dei dati '{folder_path}' non è stata trovata.")
        return pd.DataFrame()

    if not files:
        print(f"Attenzione: Nessun file CSV trovato per l'anno {year} nella cartella {folder_path}")
        return pd.DataFrame()

    dfs = []
    column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    for file in files:
        full_path = os.path.join(folder_path, file)
        df_pd_chunk = None
        try:
            df_pd_chunk = pd.read_csv(
                 full_path, header=None, names=column_names, engine='c'
            )
            df_pd_chunk['Datetime'] = pd.to_datetime(df_pd_chunk['Date'] + ' ' + df_pd_chunk['Time'], format="%Y.%m.%d %H:%M")
            df_pd_chunk = df_pd_chunk.drop(columns=['Date', 'Time'])
        except Exception as e_c:
            print(f"Info: Lettura con engine 'c' fallita per {full_path}: {e_c}. Tento con engine 'python'.")
            try:
                 df_pd_chunk = pd.read_csv(
                     full_path, header=None, names=column_names, engine='python'
                 )
                 df_pd_chunk['Datetime'] = pd.to_datetime(df_pd_chunk['Date'] + ' ' + df_pd_chunk['Time'], format="%Y.%m.%d %H:%M")
                 df_pd_chunk = df_pd_chunk.drop(columns=['Date', 'Time'])
            except Exception as e_py:
                 print(f"ERRORE: Lettura fallita anche con engine 'python' per {full_path}: {e_py}")
                 continue

        if df_pd_chunk is not None:
            try:
                df_pd_chunk = df_pd_chunk.set_index('Datetime')
                dfs.append(df_pd_chunk[['Open', 'High', 'Low', 'Close']])
            except KeyError as ke:
                 print(f"ERRORE: Colonna 'Datetime' o OHLC non trovata dopo la lettura di {full_path}: {ke}. Salto il file.")
                 continue
            except Exception as post_e:
                 print(f"ERRORE: Errore post-lettura per {full_path}: {post_e}. Salto il file.")
                 continue

    if not dfs:
         print(f"Nessun DataFrame caricato con successo per l'anno {year}.")
         return pd.DataFrame()

    full_df_pd = pd.concat(dfs).sort_index()
    if full_df_pd.index.tz is not None:
        full_df_pd.index = full_df_pd.index.tz_localize(None)
    full_df_pd = full_df_pd[~full_df_pd.index.duplicated(keep='first')]
    print(f"Dati caricati per l'anno {year}: {len(full_df_pd)} righe.")
    return full_df_pd

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
    nan_count_rsi = 14 - 1
    if len(rsi) > nan_count_rsi:
        rsi[:nan_count_rsi] = np.nan
    try:
        for pattern in talib.get_function_groups()["Pattern Recognition"]:
            result = getattr(talib, pattern)(open_, high, low, close)
            bullish |= result > 0
            bearish |= result < 0
    except Exception as e:
        print(f"Errore durante il calcolo dei pattern TA-Lib: {e}")
    return rsi, bullish, bearish

def generate_signals(close, rsi, bullish, bearish, params):
    upper, _, lower = talib.BBANDS(close, 14, params['bb_std'], params['bb_std'])
    nan_count_bb = 14 - 1
    valid_signals = (np.arange(len(close)) >= nan_count_bb) & ~np.isnan(rsi) & ~np.isnan(upper) & ~np.isnan(lower)
    entries_long = np.zeros_like(close, dtype=bool)
    exits_long = np.zeros_like(close, dtype=bool)
    entries_short = np.zeros_like(close, dtype=bool)
    exits_short = np.zeros_like(close, dtype=bool)
    entries_long[valid_signals] = (rsi[valid_signals] < params['rsi_entry']) & (close[valid_signals] < lower[valid_signals]) & bullish[valid_signals]
    exits_long[valid_signals] = (rsi[valid_signals] > params['rsi_exit']) & (close[valid_signals] > upper[valid_signals]) & bearish[valid_signals]
    entries_short[valid_signals] = (rsi[valid_signals] > params['rsi_exit']) & (close[valid_signals] > upper[valid_signals]) & bearish[valid_signals]
    exits_short[valid_signals] = (rsi[valid_signals] < params['rsi_entry']) & (close[valid_signals] < lower[valid_signals]) & bullish[valid_signals]
    return entries_long, exits_long, entries_short, exits_short

def backtest(df, indicators, params):
    # Assicurati che questa funzione sia self-contained e non modifichi variabili globali
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    datetime_index = df.index
    n_steps = len(close)
    rsi, bullish, bearish = indicators
    entries_long, exits_long, entries_short, exits_short = generate_signals(close, rsi, bullish, bearish, params)
    atr = talib.ATR(high, low, close, params['atr_window'])
    nan_count_atr = params['atr_window'] - 1
    if len(atr) > nan_count_atr:
         atr[:nan_count_atr] = np.nan
    cash = float(INITIAL_CASH)
    position = None
    equity_values = np.full(n_steps, np.nan, dtype=float)
    for i in range(n_steps):
        if np.isnan(rsi[i]) or np.isnan(atr[i]):
            equity_values[i] = INITIAL_CASH if i == 0 or np.isnan(equity_values[i-1]) else equity_values[i-1]
            continue
        price = close[i]
        unrealized_pnl = 0.0
        if position:
            unrealized_pnl = (price - position['price'] if position['direction'] == 'LONG' else position['price'] - price) * position['size']
        current_equity = cash + unrealized_pnl
        equity_values[i] = current_equity
        if position:
            exit_reason = None
            pnl = 0.0
            if position['direction'] == 'LONG':
                if price <= position['sl']: exit_reason = 'Stop Loss'
                elif price >= position['tp']: exit_reason = 'Take Profit'
                elif exits_long[i]: exit_reason = 'Signal Exit'
            elif position['direction'] == 'SHORT':
                if price >= position['sl']: exit_reason = 'Stop Loss'
                elif price <= position['tp']: exit_reason = 'Take Profit'
                elif exits_short[i]: exit_reason = 'Signal Exit'
            if exit_reason:
                pnl = (price - position['price'] if position['direction'] == 'LONG' else position['price'] - price) * position['size']
                cash += pnl
                equity_values[i] = cash
                position = None
        if not position:
            entry_made = False
            direction = None
            sl, tp = np.nan, np.nan
            if entries_long[i]:
                direction = 'LONG'
                sl = price - atr[i] * params['atr_factor']
                tp = price + atr[i] * params['atr_factor'] * 2
                entry_made = True
            elif entries_short[i]:
                direction = 'SHORT'
                sl = price + atr[i] * params['atr_factor']
                tp = price - atr[i] * params['atr_factor'] * 2
                entry_made = True
            if entry_made:
                size_equity_base = equity_values[i]
                if np.isnan(size_equity_base): size_equity_base = cash
                size_eur = size_equity_base * params['exposure']
                size = (size_eur * LEVERAGE) / price
                if size > 0:
                    position = {'price': price, 'size': size, 'sl': sl, 'tp': tp, 'time': datetime_index[i], 'direction': direction}
    first_valid_idx = np.argmax(~np.isnan(equity_values))
    if np.isnan(equity_values[first_valid_idx]):
         equity_values.fill(INITIAL_CASH)
    else:
         equity_values[:first_valid_idx] = INITIAL_CASH
         temp_series = pd.Series(equity_values)
         temp_series.ffill(inplace=True)
         equity_values = temp_series.to_numpy()
    equity_curve = pd.Series(equity_values, index=datetime_index, dtype=float)
    returns = equity_curve.pct_change().fillna(0.0)
    if returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    if returns.eq(0).all() or returns.isnull().all():
        return None
    return returns

def find_resume_point_from_partials(partial_folder, save_every):
    last_year = None
    max_part_id = -1
    file_pattern = re.compile(r"pyfolio_results_(\d{4})_part_(\d+)\.csv")
    parsed_files = []
    try:
        for filename in os.listdir(partial_folder):
            match = file_pattern.match(filename)
            if match:
                year = int(match.group(1))
                part_id = int(match.group(2))
                parsed_files.append({'year': year, 'part_id': part_id})
    except FileNotFoundError:
        print(f"Info: Cartella risultati parziali '{partial_folder}' non trovata. Si parte dall'inizio.")
        return None, -1
    if not parsed_files:
        print("Info: Nessun file parziale trovato. Si parte dall'inizio.")
        return None, -1
    last_year = max(f['year'] for f in parsed_files)
    max_part_id = max(f['part_id'] for f in parsed_files if f['year'] == last_year)
    last_index = (max_part_id * save_every) - 1
    print(f"Trovato ultimo stato dai file parziali: Anno={last_year}, Ultimo PartID={max_part_id}, Ultimo Indice Salvato={last_index}")
    return last_year, last_index

def run_backtest_worker(args, df_ref, indicators_ref, year_ref):
    idx, params = args
    try:
        returns = backtest(df_ref, indicators_ref, params)
        if returns is not None and not returns.empty:
            try:
                perf_stats = pf.timeseries.perf_stats(returns)
                stats_dict = {'year': year_ref, **params, **perf_stats.to_dict()}
                return idx, stats_dict
            except Exception as pf_err:
                 return idx, None
        else:
            return idx, None
    except Exception as e:
        return idx, None

# --- Blocco Principale ---
if __name__ == '__main__':
    # --- Gestione Argomenti Command Line ---
    parser = argparse.ArgumentParser(description="Esegue backtesting parallelo di strategie Forex.")
    parser.add_argument(
        '--years',
        metavar='Y',
        type=int,
        nargs='+',  # Accetta uno o più anni
        required=True, # Rende l'argomento obbligatorio
        help='Lista degli anni da processare (es. --years 2013 2014 2015)'
    )
    args = parser.parse_args()
    years_to_process = sorted(list(set(args.years))) # Ordina e rimuovi duplicati
    # -----------------------------------------

    script_dir = os.path.dirname(__file__)
    partial_results_path = os.path.join(script_dir, PARTIAL_RESULTS_FOLDER)
    os.makedirs(partial_results_path, exist_ok=True)

    resume_year, resume_idx = find_resume_point_from_partials(partial_results_path, SAVE_EVERY)

    combinations = generate_combinations(PARAM_GRID)
    total_combinations = len(combinations)
    print(f"Anni da processare: {years_to_process}")
    print(f"Numero totale di combinazioni da testare per anno: {total_combinations}")

    all_stats_results = []

    start_year_index = 0
    if resume_year is not None:
        # Verifica se l'anno di ripresa è nella lista fornita dall'utente
        if resume_year in years_to_process:
            try:
                start_year_index = years_to_process.index(resume_year)
            except ValueError: # Non dovrebbe succedere data la verifica precedente
                 print(f"Attenzione: Errore interno nel trovare l'indice dell'anno di ripresa {resume_year}. Si parte dall'inizio.")
                 resume_year = None
                 resume_idx = -1
        else:
            print(f"Attenzione: Anno di ripresa {resume_year} (dai file parziali) non presente negli anni richiesti ({years_to_process}). Si parte dall'inizio degli anni richiesti.")
            resume_year = None # Non riprendere da un anno non richiesto
            resume_idx = -1

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Utilizzo di {num_processes} processi worker.")

    # Modifica: Cicla sugli anni forniti dall'utente, partendo dall'indice calcolato
    for year_idx, year in enumerate(years_to_process[start_year_index:], start=start_year_index):
        main_loop_year_start_time = time.time()
        print(f"\n===== {'Ripresa' if year == resume_year else 'Inizio'} elaborazione anno: {year} =====")

        try:
            df_pd = load_forex_data(year)
            min_len_required = max(PARAM_GRID.get('atr_window', [14])) + 14
            if df_pd.empty or len(df_pd) < min_len_required:
                print(f"Skipping anno {year} (dati mancanti/insufficienti).")
                continue
            print("Calcolo indicatori per l'anno...")
            indicators = calculate_indicators(df_pd)
            print("Indicatori calcolati.")
        except Exception as e:
            print(f"Errore preparazione dati per anno {year}: {e}. Skipping.")
            continue

        processed_in_session_count = 0
        batch_stats_results = []

        start_comb_idx = 0
        part_id = 1
        if year == resume_year:
            start_comb_idx = resume_idx + 1
            part_id = (resume_idx // SAVE_EVERY) + 1
            if (resume_idx + 1) % SAVE_EVERY == 0:
                 part_id += 1
            print(f"Riprendendo anno {year} dalla combinazione {start_comb_idx + 1}, prossimo file parziale sarà part_{part_id}")
        else:
             print(f"Iniziando anno {year} dalla combinazione 1, prossimo file parziale sarà part_1")

        if start_comb_idx >= total_combinations:
             print(f"Anno {year} già completato. Salto.")
             continue

        tasks_args = [(idx, combinations[idx]) for idx in range(start_comb_idx, total_combinations)]
        num_tasks = len(tasks_args)
        print(f"Avvio multiprocessing per {num_tasks} combinazioni rimanenti per l'anno {year}...")

        results_processed_count = 0
        last_print_time = time.time()

        worker_func_partial = partial(run_backtest_worker, df_ref=df_pd, indicators_ref=indicators, year_ref=year)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(worker_func_partial, tasks_args)

            for original_idx, stats_dict in results_iterator:
                results_processed_count += 1
                current_time = time.time()

                if stats_dict is not None:
                    all_stats_results.append(stats_dict)
                    batch_stats_results.append(stats_dict)
                    processed_in_session_count += 1

                    if len(batch_stats_results) >= SAVE_EVERY:
                        partial_df = pd.DataFrame(batch_stats_results)
                        partial_filename = f"pyfolio_results_{year}_part_{part_id}.csv"
                        partial_output_path = os.path.join(partial_results_path, partial_filename)
                        try:
                            partial_df.to_csv(partial_output_path, index=False, mode='w')
                            print(f"\n✅ Batch parziale salvato: {partial_output_path} ({len(batch_stats_results)} risultati)")
                            batch_stats_results = []
                            part_id += 1
                        except Exception as e:
                            print(f"\nERRORE salvataggio file parziale {partial_output_path}: {e}")
                        last_print_time = current_time

                if results_processed_count % 50 == 0 or current_time - last_print_time > 10:
                    elapsed_time = current_time - main_loop_year_start_time
                    avg_time = elapsed_time / results_processed_count if results_processed_count > 0 else 0
                    print(f"\r  Anno {year}: Processati {results_processed_count}/{num_tasks} ({(results_processed_count/num_tasks)*100:.1f}%) | Validi: {processed_in_session_count} | Avg time/task: {avg_time:.3f}s", end="")
                    last_print_time = current_time

        print(f"\r  Anno {year}: Processati {results_processed_count}/{num_tasks} (100.0%) | Validi: {processed_in_session_count} | Tempo totale anno: {time.time() - main_loop_year_start_time:.2f}s")

        if batch_stats_results:
            partial_df = pd.DataFrame(batch_stats_results)
            partial_filename = f"pyfolio_results_{year}_part_{part_id}.csv"
            partial_output_path = os.path.join(partial_results_path, partial_filename)
            try:
                partial_df.to_csv(partial_output_path, index=False, mode='w')
                print(f"✅ Batch parziale finale salvato: {partial_output_path} ({len(batch_stats_results)} risultati)")
            except Exception as e:
                print(f"ERRORE salvataggio file parziale finale {partial_output_path}: {e}")

        print(f"--- Anno {year} completato in multiprocessing. Statistiche valide calcolate (sessione): {processed_in_session_count}. ---")

    print("\n===== Salvataggio riepilogo statistiche Pyfolio (sessione corrente) =====")
    if all_stats_results:
        summary_df = pd.DataFrame(all_stats_results)
        summary_output_path = os.path.join(script_dir, SUMMARY_RESULTS_FILE)
        try:
            summary_df.to_csv(summary_output_path, index=False, mode='w')
            print(f"✅ Riepilogo (solo sessione corrente) salvato: {summary_output_path} ({len(all_stats_results)} risultati)")
        except Exception as e:
            print(f"ERRORE salvataggio file riepilogo {summary_output_path}: {e}")
    else:
        print("Nessuna statistica valida generata in questa sessione.")

    print("\n🏁 Elaborazione completata.")
