import os
import polars as pl
import numpy as np
import talib
import itertools
import pandas as pd
import glob # Import glob
import concurrent.futures # Importa concurrent.futures
import math # Importa math

INITIAL_CASH = 1000
LEVERAGE = 100
# SAVE_EVERY = 100 # Non più necessario con questo approccio parallelo
FOLDER = '../dati_forex/EURUSD/'
SAVE_BATCH_SIZE = 5000 # Salva ogni 1000 combinazioni completate (aggiustabile)

PARAM_GRID = {
    "rsi_entry": list(range(30, 46)),
    "rsi_exit": list(range(55, 71)),
    "bb_std": [1.5, 1.75, 2.0],
    "exposure": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "atr_window": [14, 20],
    "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
}

YEARS_INPUT = [2013, 2014, 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]

def load_forex_data(year):
    files = sorted([f for f in os.listdir(FOLDER) if f.endswith('.csv') and str(year) in f])
    dfs = []
    for file in files:
        try: # Aggiunto try-except per file corrotti o vuoti
            df = pl.read_csv(
                os.path.join(FOLDER, file), has_header=False,
                new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close']
            ).with_columns(
                pl.concat_str(["Date", pl.lit(" "), "Time"])
                .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M")
                .alias("Datetime")
            ).select(["Datetime", "Open", "High", "Low", "Close"])
            if not df.is_empty():
                dfs.append(df)
            else:
                print(f"Attenzione: il file {file} è vuoto.")
        except Exception as e:
            print(f"Errore durante la lettura del file {file}: {e}")

    if not dfs:
        return pl.DataFrame() # Restituisce un DataFrame vuoto se nessun file è stato letto correttamente

    concatenated_df = pl.concat(dfs).sort("Datetime")
    concatenated_df = concatenated_df.unique(subset=["Datetime"], keep="first")
    return concatenated_df


def generate_combinations(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def calculate_indicators(df):
    # Assicurati che df non sia vuoto
    if df.is_empty():
        raise ValueError("Il DataFrame è vuoto, impossibile calcolare gli indicatori.")
    close = df["Close"].to_numpy()
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    rsi = talib.RSI(close, 14)
    bullish, bearish = np.zeros(len(close), bool), np.zeros(len(close), bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        try: # Aggiunto try-except per gestire eventuali errori nelle funzioni TA-Lib
            result = getattr(talib, pattern)(open_, high, low, close)
            # Gestisci output che potrebbero non essere della stessa lunghezza (raro ma possibile)
            if len(result) == len(close):
                 bullish |= result > 0
                 bearish |= result < 0
            else:
                 print(f"Attenzione: output di lunghezza diversa per {pattern}. Ignorato.")
        except Exception as e:
            print(f"Errore nel calcolo del pattern {pattern}: {e}")
            # Puoi decidere come gestire l'errore, ad esempio continuando
            continue
    return rsi, bullish, bearish

def generate_signals(close, rsi, bullish, bearish, params):
    upper, _, lower = talib.BBANDS(close, 14, params['bb_std'], params['bb_std'])

    # Aggiungi controllo per NaN in indicatori (es. all'inizio della serie)
    valid_indices = ~np.isnan(rsi) & ~np.isnan(upper) & ~np.isnan(lower)

    entries_long = np.zeros_like(close, dtype=bool)
    exits_long = np.zeros_like(close, dtype=bool)
    entries_short = np.zeros_like(close, dtype=bool)
    exits_short = np.zeros_like(close, dtype=bool)

    entries_long[valid_indices] = (rsi[valid_indices] < params['rsi_entry']) & (close[valid_indices] < lower[valid_indices]) & bullish[valid_indices]
    exits_long[valid_indices] = (rsi[valid_indices] > params['rsi_exit']) #& (close > upper) & bearish # Rimosso il filtro bearish per l'uscita long

    entries_short[valid_indices] = (rsi[valid_indices] > params['rsi_exit']) & (close[valid_indices] > upper[valid_indices]) & bearish[valid_indices]
    exits_short[valid_indices] = (rsi[valid_indices] < params['rsi_entry']) #& (close < lower) & bullish # Rimosso il filtro bullish per l'uscita short

    return entries_long, exits_long, entries_short, exits_short


def backtest(df, indicators, params, sim_id, year):
    # Assicurati che df non sia vuoto
    if df.is_empty():
        print(f"Attenzione: DataFrame vuoto passato a backtest per sim {sim_id}, anno {year}.")
        return INITIAL_CASH, []

    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    datetime = df["Datetime"].to_numpy()

    rsi, bullish, bearish = indicators
    # Verifica che gli indicatori non siano tutti NaN o vuoti
    if np.all(np.isnan(rsi)):
         print(f"Attenzione: RSI è tutto NaN per sim {sim_id}, anno {year}. Salto backtest.")
         return INITIAL_CASH, []

    entries_long, exits_long, entries_short, exits_short = generate_signals(close, rsi, bullish, bearish, params)
    atr = talib.ATR(high, low, close, params['atr_window'])

    cash, position, orders = INITIAL_CASH, None, []

    for i in range(len(close)):
        # Verifica se atr[i] è NaN o inf, in caso affermativo salta questo ciclo
        if np.isnan(atr[i]) or np.isinf(atr[i]) or atr[i] <= 0: # Aggiunto controllo per atr <= 0
            continue

        price = close[i]
        # Verifica prezzo valido
        if np.isnan(price) or price <= 0:
            continue

        if not position:
            entry_signal = False
            direction = None
            sl = np.nan
            tp = np.nan

            if i < len(entries_long) and entries_long[i]: # Aggiunto controllo indice
                direction = 'LONG'
                sl = price - atr[i] * params['atr_factor']
                tp = price + atr[i] * params['atr_factor'] * 2 # TP Ratio fisso a 2:1
                entry_signal = True
            elif i < len(entries_short) and entries_short[i]: # Aggiunto controllo indice
                direction = 'SHORT'
                sl = price + atr[i] * params['atr_factor']
                tp = price - atr[i] * params['atr_factor'] * 2 # TP Ratio fisso a 2:1
                entry_signal = True

            if entry_signal:
                # Controllo aggiuntivo per SL/TP non validi
                if direction == 'LONG' and (np.isnan(sl) or np.isnan(tp) or sl <= 0 or tp <= price or sl >= price): continue
                if direction == 'SHORT' and (np.isnan(sl) or np.isnan(tp) or tp <= 0 or sl <= price or sl <= tp): continue

                size_eur = cash * params['exposure']
                # Evita divisione per zero o prezzo non valido
                if price <= 0: continue
                size = (size_eur * LEVERAGE) / price
                if size <= 0: continue # Non aprire posizioni di size nulla o negativa

                position = {'price': price, 'size': size, 'sl': sl, 'tp': tp, 'time': datetime[i], 'direction': direction}

        elif position:
            exit_reason = None
            # Aggiunto controllo indice per exits_long/short
            can_check_exit_signal = (position['direction'] == 'LONG' and i < len(exits_long)) or \
                                    (position['direction'] == 'SHORT' and i < len(exits_short))

            if position['direction'] == 'LONG':
                if price <= position['sl']: exit_reason = 'Stop Loss'
                elif price >= position['tp']: exit_reason = 'Take Profit'
                elif can_check_exit_signal and exits_long[i]: exit_reason = 'Signal Exit'
            elif position['direction'] == 'SHORT':
                if price >= position['sl']: exit_reason = 'Stop Loss'
                elif price <= position['tp']: exit_reason = 'Take Profit'
                elif can_check_exit_signal and exits_short[i]: exit_reason = 'Signal Exit'

            if exit_reason:
                pnl = (price - position['price']) * position['size'] if position['direction'] == 'LONG' else (position['price'] - price) * position['size']
                cash += pnl

                order_data = {
                    "Simulation": sim_id, "Entry Time": position['time'], "Exit Time": datetime[i],
                    "Entry Price": position['price'], "Exit Price": price, "Size": position['size'],
                    "PnL": pnl, "Cash": cash, "Year": year, "Reason": exit_reason, **params, "Direction": position['direction']
                }

                # Controllo se il cash diventa negativo o nullo
                if cash <= 0:
                    print(f"🚨 Simulazione {sim_id} (Anno {year}) terminata: Cash esaurito.")
                    order_data["Reason"] += " (Cash Esaurito)"
                    orders.append(order_data)
                    position = None
                    break # Interrompi il backtest per questa combinazione

                orders.append(order_data)
                position = None

    return cash, orders

# Rimuovi o commenta save_intermediate_results e merge_results se non più usate
# def save_intermediate_results(orders, year, part_idx):
#     """Salva i risultati intermedi."""
#     ...
#
# def merge_results(year):
#     """Unisce i file CSV intermedi per un dato anno."""
#     ...

def run_backtest_task(args):
    """Funzione wrapper per eseguire backtest in un processo separato."""
    df, indicators, params, idx, year = args
    # Nota: df e indicators vengono passati per copia (serializzati) ad ogni processo.
    try:
        final_cash, orders = backtest(df, indicators, params, idx, year)
        # Stampa meno frequente per non intasare l'output
        # if idx % 100 == 0: # Stampa ogni 100 combinazioni completate (circa)
        #    print(f"  ... Anno {year} | Combinazione {idx} completata.")
        return orders
    except Exception as e:
        # Stampa l'errore e i parametri che l'hanno causato
        print(f"\n❌ Errore Anno {year} | Combinazione {idx} | Params: {params}")
        import traceback
        traceback.print_exc() # Stampa il traceback completo dell'errore
        return [] # Restituisce una lista vuota in caso di errore


if __name__ == '__main__':
    combinations = generate_combinations(PARAM_GRID)
    total_combinations = len(combinations)
    num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    print(f"⚙️ Parametri: {total_combinations} combinazioni totali.")
    print(f"🚀 Utilizzo di {num_workers} worker per il multiprocessing.")    

    all_years_orders_aggregated = [] # Lista per l'aggregazione finale

    for year in YEARS_INPUT:
        print(f"\n===== Elaborazione Anno: {year} =====")
        try:
            print(f"⏳ Caricamento dati per l'anno {year}...")
            df = load_forex_data(year)
            if df.is_empty():
                print(f"🟡 Nessun dato caricato o tutti i file erano vuoti/corrotti per l'anno {year}. Salto...")
                continue
            print(f"✅ Dati caricati: {len(df)} righe.")
            df = df.rechunk()
            print(f"📊 Calcolo indicatori per l'anno {year}...")
            indicators = calculate_indicators(df)
            print(f"✅ Indicatori calcolati.")
        except Exception as e:
            print(f"❌ Errore critico durante il caricamento dati o calcolo indicatori per l'anno {year}: {e}")
            import traceback
            traceback.print_exc()
            continue

        tasks = []
        for idx, params in enumerate(combinations, 1):
            task_args = (df, indicators, params, idx, year)
            tasks.append(task_args)

        # Prepara il file di output per l'anno corrente
        folder = f'orders/sim_short/{year}'
        os.makedirs(folder, exist_ok=True)
        final_file_path = f'{folder}/orders_{year}_merged.csv'
        # Rimuovi il file se esiste già per evitare accodamenti da run precedenti
        if os.path.exists(final_file_path):
            os.remove(final_file_path)

        print(f"⏳ Avvio backtesting parallelo per {len(tasks)} combinazioni sull'anno {year}...")
        print(f"💾 I risultati verranno salvati in batch su: {final_file_path}")

        completed_tasks = 0
        orders_batch = [] # Lista per raccogliere i risultati prima del salvataggio
        header_written = False # Flag per controllare se l'header è stato scritto

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results_iterator = executor.map(run_backtest_task, tasks)

            for orders_list in results_iterator:
                if orders_list:
                    orders_batch.extend(orders_list)
                completed_tasks += 1

                # Salva il batch se raggiunge la dimensione definita o se è l'ultimo task
                # Il controllo len(orders_batch) > 0 evita salvataggi vuoti
                if len(orders_batch) > 0 and (completed_tasks % SAVE_BATCH_SIZE == 0 or completed_tasks == total_combinations):
                    try:
                        # Converti il batch in DataFrame Pandas
                        batch_df = pd.DataFrame(orders_batch)
                        # Assicura tipi corretti (potrebbe non essere necessario se i tipi sono già corretti)
                        batch_df['Entry Time'] = pd.to_datetime(batch_df['Entry Time'])
                        batch_df['Exit Time'] = pd.to_datetime(batch_df['Exit Time'])
                        # Ordina il batch (opzionale, ma mantiene l'ordine nel file parziale)
                        batch_df = batch_df.sort_values(by='Entry Time').reset_index(drop=True)

                        # Scrivi su CSV: scrivi l'header solo la prima volta, poi appende
                        if not header_written:
                            batch_df.to_csv(final_file_path, index=False, mode='w', header=True)
                            header_written = True
                        else:
                            batch_df.to_csv(final_file_path, index=False, mode='a', header=False)

                        print(f"\r💾 Anno {year} | Batch salvato ({len(orders_batch)} ordini). Completate: {completed_tasks}/{total_combinations} ({completed_tasks/total_combinations:.1%})", end="")
                        # Aggiungi gli ordini del batch alla lista aggregata generale (se vuoi ancora l'aggregato finale in memoria)
                        all_years_orders_aggregated.extend(orders_batch)
                        orders_batch = [] # Svuota il batch dopo il salvataggio

                    except Exception as e:
                        print(f"\n❌ Errore durante il salvataggio del batch per l'anno {year}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Decidi se continuare o interrompere in caso di errore di salvataggio
                else:
                     # Aggiorna solo il progresso se non si salva
                     print(f"\r🔄 Anno {year} | Completate: {completed_tasks}/{total_combinations} ({completed_tasks/total_combinations:.1%})", end="")


        print(f"\n🏁 Backtesting parallelo per l'anno {year} completato.")
        if os.path.exists(final_file_path):
             print(f"✅ File CSV finale per l'anno {year} salvato: {final_file_path}")
        else:
             print(f"🟡 Nessun ordine generato o salvato per l'anno {year}.")


    print("\n===== Elaborazione Totale Completata =====")

    # L'aggregazione finale ora può essere fatta leggendo i file CSV per anno,
    # oppure usando la lista all_years_orders_aggregated se la memoria lo permette
    # e se è stata popolata durante il salvataggio dei batch.

    # Esempio usando la lista in memoria (se non troppo grande):
    if all_years_orders_aggregated:
        print(f"📊 Totale ordini generati (in memoria): {len(all_years_orders_aggregated)}")
        try:
            all_results_df = pd.DataFrame(all_years_orders_aggregated)
            # Ri-assicurati che i tipi datetime siano corretti e ordina
            all_results_df['Entry Time'] = pd.to_datetime(all_results_df['Entry Time'])
            all_results_df['Exit Time'] = pd.to_datetime(all_results_df['Exit Time'])
            all_results_df = all_results_df.sort_values(by=['Year', 'Entry Time']).reset_index(drop=True)

            aggregated_folder = 'orders/sim_short/aggregated'
            os.makedirs(aggregated_folder, exist_ok=True)
            aggregated_file_path = f'{aggregated_folder}/orders_all_years_aggregated_from_memory.csv'
            all_results_df.to_csv(aggregated_file_path, index=False)
            print(f"✅ File CSV aggregato (da memoria) salvato: {aggregated_file_path}")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio del file CSV aggregato (da memoria): {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🟡 Nessun ordine raccolto in memoria per l'aggregazione finale.")

    # Alternativa: Aggregazione leggendo i file CSV annuali (più robusta per memoria)
    print("\n💾 Tentativo di aggregazione leggendo i file CSV annuali...")
    all_year_files = glob.glob('orders/sim_short/*/*.csv') # Trova tutti i file CSV generati per anno
    aggregated_dfs = []
    for f in all_year_files:
        try:
            # Leggi con Pandas, assumendo che i tipi siano corretti nel CSV
            # Potrebbe essere necessario specificare i tipi se ci sono problemi
            df_year = pd.read_csv(f, parse_dates=['Entry Time', 'Exit Time'])
            aggregated_dfs.append(df_year)
        except Exception as e:
            print(f"❌ Errore durante la lettura del file {f} per l'aggregazione: {e}")

    if aggregated_dfs:
        try:
            final_aggregated_df = pd.concat(aggregated_dfs, ignore_index=True)
            final_aggregated_df = final_aggregated_df.sort_values(by=['Year', 'Entry Time']).reset_index(drop=True)

            aggregated_folder = 'orders/sim_short/aggregated'
            os.makedirs(aggregated_folder, exist_ok=True)
            aggregated_file_path_from_files = f'{aggregated_folder}/orders_all_years_aggregated_from_files.csv'
            final_aggregated_df.to_csv(aggregated_file_path_from_files, index=False)
            print(f"✅ File CSV aggregato (da file annuali) salvato: {aggregated_file_path_from_files} ({len(final_aggregated_df)} ordini)")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio del file CSV aggregato (da file annuali): {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🟡 Nessun file annuale trovato o letto correttamente per l'aggregazione.")
