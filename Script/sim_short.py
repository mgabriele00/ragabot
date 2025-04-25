import os
import polars as pl
import numpy as np
import talib
import itertools
import pandas as pd
import glob # Import glob

INITIAL_CASH = 1000
LEVERAGE = 100
SAVE_EVERY = 100 # Salva ogni 200 combinazioni
FOLDER = '../dati_forex/EURUSD/'

PARAM_GRID = {
    "rsi_entry": list(range(30, 46)),
    "rsi_exit": list(range(55, 71)),
    "bb_std": [1.5, 1.75, 2.0],
    "exposure": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "atr_window": [14, 20],
    "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
}

YEARS_INPUT = [2013, 2014, 2015]

def load_forex_data(year):
    files = sorted([f for f in os.listdir(FOLDER) if f.endswith('.csv') and str(year) in f])
    dfs = []
    for file in files:
        df = pl.read_csv(
            os.path.join(FOLDER, file), has_header=False,
            new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close']
        ).with_columns(
            pl.concat_str(["Date", pl.lit(" "), "Time"])
            .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M")
            .alias("Datetime")
        ).select(["Datetime", "Open", "High", "Low", "Close"])
        dfs.append(df)
    # Gestione dei duplicati prima della concatenazione se necessario
    # df = df.unique(subset=["Datetime"], keep="first")
    concatenated_df = pl.concat(dfs).sort("Datetime")
    # Rimuovi eventuali duplicati dopo la concatenazione
    concatenated_df = concatenated_df.unique(subset=["Datetime"], keep="first")
    return concatenated_df


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
        try: # Aggiunto try-except per gestire eventuali errori nelle funzioni TA-Lib
            result = getattr(talib, pattern)(open_, high, low, close)
            bullish |= result > 0
            bearish |= result < 0
        except Exception as e:
            print(f"Errore nel calcolo del pattern {pattern}: {e}")
            # Puoi decidere come gestire l'errore, ad esempio continuando
            continue
    return rsi, bullish, bearish

def generate_signals(close, rsi, bullish, bearish, params):
    upper, _, lower = talib.BBANDS(close, 14, params['bb_std'], params['bb_std'])

    entries_long = (rsi < params['rsi_entry']) & (close < lower) & bullish
    exits_long = (rsi > params['rsi_exit']) #& (close > upper) & bearish # Rimosso il filtro bearish per l'uscita long

    entries_short = (rsi > params['rsi_exit']) & (close > upper) & bearish
    exits_short = (rsi < params['rsi_entry']) #& (close < lower) & bullish # Rimosso il filtro bullish per l'uscita short

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
        # Verifica se atr[i] è NaN, in caso affermativo salta questo ciclo
        if np.isnan(atr[i]):
            continue

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

            # Controllo aggiuntivo per SL/TP non validi (es. negativi o troppo vicini)
            if direction == 'LONG' and (sl <= 0 or tp <= price): continue
            if direction == 'SHORT' and (tp <= 0 or sl <= price): continue


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
                # Controllo se il cash diventa negativo o nullo
                if cash <= 0:
                    print(f"🚨 Simulazione {sim_id} terminata: Cash esaurito.")
                    # Aggiungi l'ultimo ordine anche se il cash è <= 0 per tracciabilità
                    orders.append({
                        "Simulation": sim_id, "Entry Time": position['time'], "Exit Time": datetime[i],
                        "Entry Price": position['price'], "Exit Price": price, "Size": position['size'],
                        "PnL": pnl, "Cash": cash, "Year": year, "Reason": exit_reason + " (Cash Esaurito)", **params, "Direction": position['direction']
                    })
                    position = None
                    break # Interrompi il backtest per questa combinazione

                orders.append({
                    "Simulation": sim_id, "Entry Time": position['time'], "Exit Time": datetime[i],
                    "Entry Price": position['price'], "Exit Price": price, "Size": position['size'],
                    "PnL": pnl, "Cash": cash, "Year": year, "Reason": exit_reason, **params, "Direction": position['direction']
                })
                position = None

    # Se c'è una posizione aperta alla fine, non la chiudiamo (o potremmo decidere di chiuderla all'ultimo prezzo)
    # Al momento, viene semplicemente ignorata se non chiusa da SL/TP/Signal

    return cash, orders

def save_intermediate_results(orders, year, part_idx):
    """Salva i risultati intermedi."""
    if not orders: return # Non salvare se non ci sono ordini
    folder = f'orders/sim_short/{year}'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/orders_{year}_part_{part_idx}.csv'
    pd.DataFrame(orders).to_csv(file_path, index=False)
    print(f"💾 CSV intermedio salvato: {file_path}")

def merge_results(year):
    """Unisce i file CSV intermedi per un dato anno."""
    folder = f'orders/sim_short/{year}'
    intermediate_files = glob.glob(f'{folder}/orders_{year}_part_*.csv')
    if not intermediate_files:
        print(f"Nessun file intermedio trovato per l'anno {year} in {folder}.")
        return

    all_dfs = []
    for f in intermediate_files:
        try:
            df_part = pd.read_csv(f)
            all_dfs.append(df_part)
        except pd.errors.EmptyDataError:
            print(f"Attenzione: il file {f} è vuoto e sarà ignorato.")
        except Exception as e:
            print(f"Errore durante la lettura di {f}: {e}")


    if not all_dfs:
        print(f"Nessun dato valido trovato nei file intermedi per l'anno {year}.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    final_file_path = f'{folder}/orders_{year}_merged.csv'
    merged_df.to_csv(final_file_path, index=False)
    print(f"✅ File CSV unito salvato: {final_file_path}")

    # Opzionale: rimuovere i file intermedi dopo l'unione
    # for f in intermediate_files:
    #     try:
    #         os.remove(f)
    #         print(f"🗑️ File intermedio rimosso: {f}")
    #     except OSError as e:
    #         print(f"Errore nella rimozione del file {f}: {e}")


if __name__ == '__main__':
    combinations = generate_combinations(PARAM_GRID)
    total_combinations = len(combinations)

    for year in YEARS_INPUT:
        print(f"\n===== Elaborazione Anno: {year} =====")
        df = load_forex_data(year)
        if df.is_empty():
            print(f"Nessun dato caricato per l'anno {year}. Salto...")
            continue

        indicators = calculate_indicators(df)
        all_orders_chunk = []
        part_idx = 1

        for idx, params in enumerate(combinations, 1):
            # Calcola l'ID univoco della simulazione globale (opzionale, ma utile se si uniscono anni diversi)
            # global_sim_id = (year - min(YEARS_INPUT)) * total_combinations + idx
            print(f"\r🔄 Anno {year} | Combinazione {idx}/{total_combinations}...", end="") # Aggiunto \r e end="" per aggiornare sulla stessa riga

            # Passa l'indice 'idx' come sim_id per mantenere la coerenza con l'output precedente
            final_cash, orders = backtest(df, indicators, params, idx, year)
            all_orders_chunk.extend(orders)

            # Salva ogni SAVE_EVERY combinazioni
            if idx % SAVE_EVERY == 0:
                save_intermediate_results(all_orders_chunk, year, part_idx)
                all_orders_chunk = [] # Resetta la lista per il prossimo chunk
                part_idx += 1

        # Salva l'ultimo chunk se non è vuoto e non è stato già salvato
        if all_orders_chunk:
            save_intermediate_results(all_orders_chunk, year, part_idx)

        print(f"\n🏁 Elaborazione combinazioni per l'anno {year} completata.")

        # Unisci i risultati per l'anno corrente
        print(f"🤝 Unione dei risultati per l'anno {year}...")
        merge_results(year)

    print("\n===== Elaborazione Completata =====")
