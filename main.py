import polars as pl
import numpy as np
import talib
import pickle
import os
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union

# === Parametri base ===
INITIAL_CASH = 1000
LEVERAGE = 100
SAVE_EVERY = 200  # salva ogni N combinazioni

def load_forex_data(folder_path: str) -> pl.DataFrame:
    """Carica e prepara i dati Forex dai file CSV utilizzando Polars"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        # Leggi il CSV con Polars
        temp_df = pl.read_csv(file_path, has_header=False)
        # Rinomina le colonne
        temp_df = temp_df.select([
            pl.col("column_1").alias("Date"),
            pl.col("column_2").alias("Time"),
            pl.col("column_3").cast(pl.Float64).alias("Open"),
            pl.col("column_4").cast(pl.Float64).alias("High"),
            pl.col("column_5").cast(pl.Float64).alias("Low"),
            pl.col("column_6").cast(pl.Float64).alias("Close")
        ])
        # Crea la colonna datetime
        temp_df = temp_df.with_columns([
            pl.concat_str([
                pl.col("Date"),
                pl.lit(" "),
                pl.col("Time")
            ]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
        ])
        df_list.append(temp_df.select(["Datetime", "Open", "High", "Low", "Close"]))

    # Concatena i dataframe e ordina per data
    return pl.concat(df_list).sort("Datetime")

def generate_parameter_combinations(parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Genera tutte le possibili combinazioni di parametri a partire da un dizionario di intervalli.
    
    Args:
        parameter_ranges: Dizionario con i nomi dei parametri come chiavi 
                         e liste di valori possibili come valori
    
    Returns:
        Lista di dizionari, ognuno rappresentante una combinazione di parametri
    """
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    return [{param_names[i]: combo[i] for i in range(len(param_names))} for combo in combinations]

def calculate_indicators(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola gli indicatori tecnici: RSI, pattern bullish e bearish
    
    Returns:
        Tuple con (timestamps, array_rsi, array_bullish, array_bearish)
    """
    # Converti le colonne Polars in array numpy
    close_np = df.select("Close").to_numpy().flatten()
    open_np = df.select("Open").to_numpy().flatten()
    high_np = df.select("High").to_numpy().flatten()
    low_np = df.select("Low").to_numpy().flatten()
    timestamps = df.select("Datetime").to_numpy().flatten()
    
    # Calcolo RSI
    rsi_np = talib.RSI(close_np, timeperiod=14)
    
    # Inizializza array booleani per pattern bullish e bearish
    bullish_np = np.zeros(len(close_np), dtype=bool)
    bearish_np = np.zeros(len(close_np), dtype=bool)
    
    # Calcolo pattern di candele
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    for name in pattern_names:
        func = getattr(talib, name)
        result = func(open_np, high_np, low_np, close_np)
        bullish_np = np.logical_or(bullish_np, result > 0)
        bearish_np = np.logical_or(bearish_np, result < 0)
    
    return timestamps, rsi_np, bullish_np, bearish_np

def generate_signals(
    close_np: np.ndarray,
    rsi_np: np.ndarray,
    bullish_np: np.ndarray, 
    bearish_np: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera i segnali di ingresso e uscita basati sugli indicatori
    
    Returns:
        Tuple con (array_entries, array_exits)
    """
    rsi_entry = params["rsi_entry"]
    rsi_exit = params["rsi_exit"]
    bb_std = params["bb_std"]
    
    # Calcola le Bande di Bollinger con array NumPy
    upper, middle, lower = talib.BBANDS(
        close_np, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std
    )
    
    # Genera segnali di ingresso e uscita (array booleani)
    entries = np.logical_and.reduce((
        rsi_np < rsi_entry,
        close_np < lower,
        bullish_np
    ))
    
    exits = np.logical_and.reduce((
        rsi_np > rsi_exit,
        close_np > upper,
        bearish_np
    ))
    
    return entries, exits

def backtest_strategy(
    df: pl.DataFrame, 
    params: Dict[str, Any], 
    timestamps: np.ndarray,
    rsi_np: np.ndarray,
    bullish_np: np.ndarray, 
    bearish_np: np.ndarray
) -> Tuple[float, List[Dict]]:
    """
    Esegue il backtest della strategia di trading con i parametri specificati
    
    Args:
        df: DataFrame con i dati di prezzo
        params: Dizionario con i parametri della strategia
        timestamps: Array NumPy con i timestamp
        rsi_np: Array NumPy con i valori RSI
        bullish_np: Array NumPy booleano dei pattern bullish
        bearish_np: Array NumPy booleano dei pattern bearish
        
    Returns:
        Tuple con (capitale_finale, lista_operazioni)
    """
    # Estrazione parametri
    sl = params["sl"]
    tp = params["tp"]
    exposure = params["exposure"]
    
    # Ottieni array numpy dai dati Polars
    close_np = df.select("Close").to_numpy().flatten()
    
    # Genera segnali
    entries, exits = generate_signals(
        close_np, rsi_np, bullish_np, bearish_np, params
    )
    
    # Converti in liste Python per il ciclo for (accesso più veloce)
    close_list = close_np.tolist()
    time_list = timestamps.tolist()
    entries_list = entries.tolist()
    exits_list = exits.tolist()
    
    # Esecuzione della strategia
    cash = INITIAL_CASH
    in_position = False
    orders = []
    
    for i in range(len(close_list)):
        price = close_list[i]
        time = time_list[i]
        
        if entries_list[i] and not in_position and cash > 0:
            size_eur = cash * exposure
            size = (size_eur * LEVERAGE) / price
            entry_price = price
            entry_time = time
            entry_size = size
            in_position = True

        elif in_position:
            if price <= entry_price * (1 - sl):
                exit_price = price
                pnl = (exit_price - entry_price) * entry_size
                cash += pnl
                orders.append({
                    'Entry Time': entry_time,
                    'Exit Time': time,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Size': entry_size,
                    'PnL': pnl,
                    'Cash': cash,
                    'SL': sl,
                    'TP': tp,
                    'RSI Entry': params['rsi_entry'],
                    'RSI Exit': params['rsi_exit'],
                    'BB Std': params['bb_std'],
                    'Exposure': exposure,
                    'Reason': 'Stop Loss'
                })
                in_position = False
                continue

            if price >= entry_price * (1 + tp):
                exit_price = price
                pnl = (exit_price - entry_price) * entry_size
                cash += pnl
                orders.append({
                    'Entry Time': entry_time,
                    'Exit Time': time,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Size': entry_size,
                    'PnL': pnl,
                    'Cash': cash,
                    'SL': sl,
                    'TP': tp,
                    'RSI Entry': params['rsi_entry'],
                    'RSI Exit': params['rsi_exit'],
                    'BB Std': params['bb_std'],
                    'Exposure': exposure,
                    'Reason': 'Take Profit'
                })
                in_position = False
                continue

            if exits_list[i]:
                exit_price = price
                pnl = (exit_price - entry_price) * entry_size
                cash += pnl
                orders.append({
                    'Entry Time': entry_time,
                    'Exit Time': time,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Size': entry_size,
                    'PnL': pnl,
                    'Cash': cash,
                    'SL': sl,
                    'TP': tp,
                    'RSI Entry': params['rsi_entry'],
                    'RSI Exit': params['rsi_exit'],
                    'BB Std': params['bb_std'],
                    'Exposure': exposure,
                    'Reason': 'Signal Exit'
                })
                in_position = False
    
    return cash, orders

def save_results(buffer_orders: List, year: int, block_id: int, is_final: bool = False,
                best_orders: Optional[List] = None) -> None:
    """Salva i risultati del backtest in file pickle, organizzati per anno"""
    if not is_final:
        # Assicurati che la directory esista
        os.makedirs(f"orders/partial/{year}", exist_ok=True)
        
        partial_file = f"orders/partial/{year}/orders_{year}_block_{block_id}.pkl"
        with open(partial_file, "wb") as f:
            pickle.dump(buffer_orders, f)
        print(f"💾 Salvato blocco intermedio: {partial_file}")
    else:
        # Assicurati che la directory esista
        os.makedirs("orders/final", exist_ok=True)
        
        final_file = f"orders/final/orders_{year}_train.pkl"
        with open(final_file, "wb") as f:
            pickle.dump(best_orders, f)
        print(f"💾 Salvato risultato finale: {final_file}")

def run_backtests(
    df_train: pl.DataFrame, 
    params_list: List[Dict[str, Any]],
    timestamps: np.ndarray,
    rsi_np: np.ndarray,
    bullish_np: np.ndarray, 
    bearish_np: np.ndarray,
    year: int,
    total_combinations: int
) -> Tuple[float, Tuple, List]:
    """
    Esegue tutti i backtest per le combinazioni di parametri
    
    Returns:
        Tuple con (miglior_risultato, migliori_parametri, migliori_ordini)
    """
    best_result = None
    best_params = None
    best_orders = None
    buffer_orders = []
    counter = 0
    
    # Determina l'ultimo blocco e il punto di partenza
    block_id, start_from = get_last_block_id(year)
    
    for param in params_list:
        counter += 1
        if counter < start_from:
            continue

        print(
            f"🔄 {counter}/{total_combinations} | SL={param['sl']}, TP={param['tp']}, "
            f"RSI_IN={param['rsi_entry']}, RSI_OUT={param['rsi_exit']}, "
            f"BB={param['bb_std']}, EXP={param['exposure']:.2f}"
        )

        # Esecuzione del backtest
        final_cash, orders = backtest_strategy(
            df_train, param, timestamps, rsi_np, bullish_np, bearish_np
        )
        buffer_orders.append((final_cash, orders))

        # Salvataggio intermedio
        if counter % SAVE_EVERY == 0:
            save_results(buffer_orders, year, block_id)
            buffer_orders = []
            block_id += 1

        # Aggiornamento dei migliori parametri
        if best_result is None or final_cash > best_result:
            best_result = final_cash
            best_params = (
                param['sl'], param['tp'], param['rsi_entry'], 
                param['rsi_exit'], param['bb_std'], param['exposure']
            )
            best_orders = orders
            
    return best_result, best_params, best_orders

def get_last_block_id(year: int) -> Tuple[int, int]:
    """
    Determina l'ultimo blocco salvato e il punto di partenza per il backtest
    
    Args:
        year: L'anno per cui cercare i blocchi salvati
    
    Returns:
        Tuple con (block_id, start_from)
    """
    # Crea le directory se non esistono
    os.makedirs(f"orders/partial/{year}", exist_ok=True)
    os.makedirs("orders/final", exist_ok=True)
    
    # Cerca tutti i file di blocco per l'anno specificato
    try:
        block_files = [f for f in os.listdir(f"orders/partial/{year}") 
                      if f.startswith(f"orders_{year}_block_") and f.endswith(".pkl")]
    except FileNotFoundError:
        # Se la directory non esiste ancora
        block_files = []
    
    if not block_files:
        # Nessun blocco trovato, iniziamo dall'inizio
        print(f"🔍 Nessun blocco precedente trovato per l'anno {year}, inizio dal principio")
        return 1, 1
    
    # Estrai i numeri di blocco dai nomi dei file
    block_numbers = []
    for file in block_files:
        try:
            block_num = int(file.split("_block_")[1].split(".")[0])
            block_numbers.append(block_num)
        except (ValueError, IndexError):
            continue
    
    if not block_numbers:
        return 1, 1
    
    # Trova l'ultimo blocco
    last_block = max(block_numbers)
    
    # Calcola il punto di partenza
    start_from = (last_block * SAVE_EVERY) + 1
    
    # Il prossimo blocco sarà last_block + 1
    next_block_id = last_block + 1
    
    print(f"🔄 Riprendo dal blocco {next_block_id}, combinazione {start_from}")
    return next_block_id, start_from

def main():
    """Funzione principale che esegue il backtesting completo"""
    # === Range di iperparametri ===
    sl_values = np.round(np.linspace(0.002, 0.010, 5), 4).tolist()
    tp_values = np.round(np.linspace(0.01, 0.03, 5), 4).tolist()
    rsi_entry_values = list(range(30, 46, 5))
    rsi_exit_values = list(range(55, 71, 5))
    bb_std_values = np.round(np.linspace(1.5, 2.5, 5), 2).tolist()
    exposures = np.round(np.linspace(0.1, 0.6, 6), 2).tolist()
    
    # === Caricamento dati ===
    folder = './dati_forex/EURUSD/'
    df = load_forex_data(folder)
    
    # === Parametri Rolling ===
    years = [2019, 2020, 2021, 2022, 2023]
    train_years_window = 1

    parameter_ranges = {
        "sl": sl_values,
        "tp": tp_values,
        "rsi_entry": rsi_entry_values,
        "rsi_exit": rsi_exit_values,
        "bb_std": bb_std_values,
        "exposure": exposures
    }
    
    # Genera tutte le combinazioni di parametri
    all_params = generate_parameter_combinations(parameter_ranges)
    total_combinations = len(all_params)
    
    # Loop principale per ciascun anno
    for year in years:
        print(f"\n🟩 Finestra: Train = {year}-{year + train_years_window - 1}, Test = {year + train_years_window}")

        # Split dei dati in train e test usando Polars
        df_train = df.filter(
            (pl.col("Datetime").dt.year() >= year) & 
            (pl.col("Datetime").dt.year() < year + train_years_window)
        )
        df_test = df.filter(pl.col("Datetime").dt.year() == year + train_years_window)

        if df_train.height == 0 or df_test.height == 0:
            print(f"⚠️ Dati mancanti per l'anno {year}, salto...")
            continue

        # Calcolo degli indicatori (una sola volta per tutti i backtest dell'anno)
        timestamps, rsi_np, bullish_np, bearish_np = calculate_indicators(df_train)
        
        # Esecuzione di tutti i backtest
        best_result, best_params, best_orders = run_backtests(
            df_train, all_params, timestamps, rsi_np, bullish_np, bearish_np, year, total_combinations
        )

        # Salvataggio finale
        save_results(None, year, 0, True, best_orders)
        print(f"✅ Train {year}: Miglior capitale = €{best_result:.2f} con parametri {best_params}")

if __name__ == "__main__":
    main()
