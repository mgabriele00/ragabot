import polars as pl
import numpy as np
import talib
import pickle
import os
import itertools
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
# === Parametri base ===
INITIAL_CASH = 1000
LEVERAGE = 100
SAVE_EVERY = 5000

def load_forex_data(folder_path: str) -> pl.DataFrame:
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        temp_df = pl.read_csv(file_path, has_header=False)
        temp_df = temp_df.select([
            pl.col("column_1").alias("Date"),
            pl.col("column_2").alias("Time"),
            pl.col("column_3").cast(pl.Float64).alias("Open"),
            pl.col("column_4").cast(pl.Float64).alias("High"),
            pl.col("column_5").cast(pl.Float64).alias("Low"),
            pl.col("column_6").cast(pl.Float64).alias("Close")
        ])
        temp_df = temp_df.with_columns([
            pl.concat_str([
                pl.col("Date"),
                pl.lit(" "),
                pl.col("Time")
            ]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
        ])
        df_list.append(temp_df.select(["Datetime", "Open", "High", "Low", "Close"]))

    return pl.concat(df_list).sort("Datetime")

def generate_parameter_combinations(parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    combinations = list(itertools.product(*param_values))
    return [{param_names[i]: combo[i] for i in range(len(param_names))} for combo in combinations]

def calculate_indicators(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close_np = df.select("Close").to_numpy().flatten()
    open_np = df.select("Open").to_numpy().flatten()
    high_np = df.select("High").to_numpy().flatten()
    low_np = df.select("Low").to_numpy().flatten()
    timestamps = df.select("Datetime").to_numpy().flatten()

    rsi_np = talib.RSI(close_np, timeperiod=14)

    bullish_np = np.zeros(len(close_np), dtype=bool)
    bearish_np = np.zeros(len(close_np), dtype=bool)

    pattern_names = talib.get_function_groups()['Pattern Recognition']
    for name in pattern_names:
        func = getattr(talib, name)
        result = func(open_np, high_np, low_np, close_np)
        bullish_np = np.logical_or(bullish_np, result > 0)
        bearish_np = np.logical_or(bearish_np, result < 0)

    return timestamps, rsi_np, bullish_np, bearish_np

def generate_signals(close_np, rsi_np, bullish_np, bearish_np, params):
    rsi_entry = params["rsi_entry"]
    rsi_exit = params["rsi_exit"]
    bb_std = params["bb_std"]

    upper, _, lower = talib.BBANDS(close_np, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)

    entries = np.logical_and.reduce((rsi_np < rsi_entry, close_np < lower, bullish_np))
    exits = np.logical_and.reduce((rsi_np > rsi_exit, close_np > upper, bearish_np))

    return entries, exits

def backtest_strategy(df, params, timestamps, rsi_np, bullish_np, bearish_np):
    sl = params["sl"]
    tp = params["tp"]
    exposure = params["exposure"]
    close_np = df.select("Close").to_numpy().flatten()
    entries, exits = generate_signals(close_np, rsi_np, bullish_np, bearish_np, params)
    close_list = close_np.tolist()
    time_list = timestamps.tolist()
    entries_list = entries.tolist()
    exits_list = exits.tolist()

    cash = INITIAL_CASH
    in_position = False
    position_type = None
    orders = []

    for i in range(len(close_list)):
        price = close_list[i]
        time = time_list[i]

        if not in_position:
            if entries_list[i]:
                size_eur = cash * exposure
                size = (size_eur * LEVERAGE) / price
                entry_price = price
                entry_time = time
                entry_size = size
                position_type = "long"
                in_position = True
            elif exits_list[i]:
                size_eur = cash * exposure
                size = (size_eur * LEVERAGE) / price
                entry_price = price
                entry_time = time
                entry_size = size
                position_type = "short"
                in_position = True

        elif in_position:
            exit_condition = False
            reason = ""
            pnl = 0

            if position_type == "long":
                if price <= entry_price * (1 - sl):
                    exit_condition = True
                    reason = "Stop Loss"
                elif price >= entry_price * (1 + tp):
                    exit_condition = True
                    reason = "Take Profit"
                elif exits_list[i]:
                    exit_condition = True
                    reason = "Signal Exit"
                if exit_condition:
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size

            elif position_type == "short":
                if price >= entry_price * (1 + sl):
                    exit_condition = True
                    reason = "Stop Loss"
                elif price <= entry_price * (1 - tp):
                    exit_condition = True
                    reason = "Take Profit"
                elif entries_list[i]:
                    exit_condition = True
                    reason = "Signal Entry"
                if exit_condition:
                    exit_price = price
                    pnl = (entry_price - exit_price) * entry_size

            if exit_condition:
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
                    'Type': position_type,
                    'Reason': reason
                })
                in_position = False
                position_type = None

    return cash, orders

def save_results(buffer_orders: List, year: int, block_id: int, is_final: bool = False,
                 best_orders: Optional[List] = None) -> None:
    if not is_final:
        os.makedirs(f"orders/partial/{year}", exist_ok=True)
        partial_file = f"orders/partial/{year}/orders_{year}_block_{block_id}.parquet"
        
        # Concatena tutti gli ordini in un unico DataFrame
        flat_orders = [order for _, orders in buffer_orders for order in orders]
        df_partial = pd.DataFrame(flat_orders)
        df_partial.to_parquet(partial_file, compression='snappy')
        print(f"💾 Salvato blocco intermedio (Parquet): {partial_file}")
        
    else:
        os.makedirs("orders/final", exist_ok=True)
        final_file = f"orders/final/orders_{year}_train.parquet"
        df_final = pd.DataFrame(best_orders)
        df_final.to_parquet(final_file, compression='snappy')
        print(f"💾 Salvato risultato finale (Parquet): {final_file}")

def get_last_block_id(year: int) -> Tuple[int, int]:
    os.makedirs(f"orders/partial/{year}", exist_ok=True)
    os.makedirs("orders/final", exist_ok=True)
    try:
        block_files = [f for f in os.listdir(f"orders/partial/{year}") 
                       if f.startswith(f"orders_{year}_block_") and f.endswith(".pkl")]
    except FileNotFoundError:
        block_files = []

    if not block_files:
        return 1, 1

    block_numbers = []
    for file in block_files:
        try:
            block_num = int(file.split("_block_")[1].split(".")[0])
            block_numbers.append(block_num)
        except (ValueError, IndexError):
            continue

    if not block_numbers:
        return 1, 1

    last_block = max(block_numbers)
    start_from = (last_block * SAVE_EVERY) + 1
    next_block_id = last_block + 1
    print(f"🔄 Riprendo dal blocco {next_block_id}, combinazione {start_from}")
    return next_block_id, start_from

def run_backtests(df_train: pl.DataFrame, params_list: List[Dict[str, Any]],
                   timestamps: np.ndarray, rsi_np: np.ndarray,
                   bullish_np: np.ndarray, bearish_np: np.ndarray,
                   year: int, total_combinations: int
                   ) -> Tuple[float, Tuple, List]:
    best_result = None
    best_params = None
    best_orders = None
    buffer_orders = []
    counter = 0

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

        final_cash, orders = backtest_strategy(
            df_train, param, timestamps, rsi_np, bullish_np, bearish_np
        )
        buffer_orders.append((final_cash, orders))

        if counter % SAVE_EVERY == 0:
            save_results(buffer_orders, year, block_id)
            buffer_orders = []
            block_id += 1

        if best_result is None or final_cash > best_result:
            best_result = final_cash
            best_params = (
                param['sl'], param['tp'], param['rsi_entrimport polars as pl
import numpy as np
import talib
import pickle
import os
import itertools
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
# === Parametri base ===
INITIAL_CASH = 1000
LEVERAGE = 100
SAVE_EVERY = 5000

def load_forex_data(folder_path: str) -> pl.DataFrame:
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        temp_df = pl.read_csv(file_path, has_header=False)
        temp_df = temp_df.select([
            pl.col("column_1").alias("Date"),
            pl.col("column_2").alias("Time"),
            pl.col("column_3").cast(pl.Float64).alias("Open"),
            pl.col("column_4").cast(pl.Float64).alias("High"),
            pl.col("column_5").cast(pl.Float64).alias("Low"),
            pl.col("column_6").cast(pl.Float64).alias("Close")
        ])
        temp_df = temp_df.with_columns([
            pl.concat_str([
                pl.col("Date"),
                pl.lit(" "),
                pl.col("Time")
            ]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
        ])
        df_list.append(temp_df.select(["Datetime", "Open", "High", "Low", "Close"]))

    return pl.concat(df_list).sort("Datetime")

def generate_parameter_combinations(parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    combinations = list(itertools.product(*param_values))
    return [{param_names[i]: combo[i] for i in range(len(param_names))} for combo in combinations]

def calculate_indicators(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close_np = df.select("Close").to_numpy().flatten()
    open_np = df.select("Open").to_numpy().flatten()
    high_np = df.select("High").to_numpy().flatten()
    low_np = df.select("Low").to_numpy().flatten()
    timestamps = df.select("Datetime").to_numpy().flatten()

    rsi_np = talib.RSI(close_np, timeperiod=14)

    bullish_np = np.zeros(len(close_np), dtype=bool)
    bearish_np = np.zeros(len(close_np), dtype=bool)

    pattern_names = talib.get_function_groups()['Pattern Recognition']
    for name in pattern_names:
        func = getattr(talib, name)
        result = func(open_np, high_np, low_np, close_np)
        bullish_np = np.logical_or(bullish_np, result > 0)
        bearish_np = np.logical_or(bearish_np, result < 0)

    return timestamps, rsi_np, bullish_np, bearish_np

def generate_signals(close_np, rsi_np, bullish_np, bearish_np, params):
    rsi_entry = params["rsi_entry"]
    rsi_exit = params["rsi_exit"]
    bb_std = params["bb_std"]

    upper, _, lower = talib.BBANDS(close_np, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)

    entries = np.logical_and.reduce((rsi_np < rsi_entry, close_np < lower, bullish_np))
    exits = np.logical_and.reduce((rsi_np > rsi_exit, close_np > upper, bearish_np))

    return entries, exits

def backtest_strategy(df, params, timestamps, rsi_np, bullish_np, bearish_np):
    sl = params["sl"]
    tp = params["tp"]
    exposure = params["exposure"]
    close_np = df.select("Close").to_numpy().flatten()
    entries, exits = generate_signals(close_np, rsi_np, bullish_np, bearish_np, params)
    close_list = close_np.tolist()
    time_list = timestamps.tolist()
    entries_list = entries.tolist()
    exits_list = exits.tolist()

    cash = INITIAL_CASH
    in_position = False
    position_type = None
    orders = []

    for i in range(len(close_list)):
        price = close_list[i]
        time = time_list[i]

        if not in_position:
            if entries_list[i]:
                size_eur = cash * exposure
                size = (size_eur * LEVERAGE) / price
                entry_price = price
                entry_time = time
                entry_size = size
                position_type = "long"
                in_position = True
            elif exits_list[i]:
                size_eur = cash * exposure
                size = (size_eur * LEVERAGE) / price
                entry_price = price
                entry_time = time
                entry_size = size
                position_type = "short"
                in_position = True

        elif in_position:
            exit_condition = False
            reason = ""
            pnl = 0

            if position_type == "long":
                if price <= entry_price * (1 - sl):
                    exit_condition = True
                    reason = "Stop Loss"
                elif price >= entry_price * (1 + tp):
                    exit_condition = True
                    reason = "Take Profit"
                elif exits_list[i]:
                    exit_condition = True
                    reason = "Signal Exit"
                if exit_condition:
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size

            elif position_type == "short":
                if price >= entry_price * (1 + sl):
                    exit_condition = True
                    reason = "Stop Loss"
                elif price <= entry_price * (1 - tp):
                    exit_condition = True
                    reason = "Take Profit"
                elif entries_list[i]:
                    exit_condition = True
                    reason = "Signal Entry"
                if exit_condition:
                    exit_price = price
                    pnl = (entry_price - exit_price) * entry_size

            if exit_condition:
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
                    'Type': position_type,
                    'Reason': reason
                })
                in_position = False
                position_type = None

    return cash, orders

def save_results(buffer_orders: List, year: int, block_id: int, is_final: bool = False,
                 best_orders: Optional[List] = None) -> None:
    if not is_final:
        os.makedirs(f"orders/partial/{year}", exist_ok=True)
        partial_file = f"orders/partial/{year}/orders_{year}_block_{block_id}.parquet"
        
        # Concatena tutti gli ordini in un unico DataFrame
        flat_orders = [order for _, orders in buffer_orders for order in orders]
        df_partial = pd.DataFrame(flat_orders)
        df_partial.to_parquet(partial_file, compression='snappy')
        print(f"💾 Salvato blocco intermedio (Parquet): {partial_file}")
        
    else:
        os.makedirs("orders/final", exist_ok=True)
        final_file = f"orders/final/orders_{year}_train.parquet"
        df_final = pd.DataFrame(best_orders)
        df_final.to_parquet(final_file, compression='snappy')
        print(f"💾 Salvato risultato finale (Parquet): {final_file}")

def get_last_block_id(year: int) -> Tuple[int, int]:
    os.makedirs(f"orders/partial/{year}", exist_ok=True)
    os.makedirs("orders/final", exist_ok=True)
    try:
        block_files = [f for f in os.listdir(f"orders/partial/{year}") 
                       if f.startswith(f"orders_{year}_block_") and f.endswith(".pkl")]
    except FileNotFoundError:
        block_files = []

    if not block_files:
        return 1, 1

    block_numbers = []
    for file in block_files:
        try:
            block_num = int(file.split("_block_")[1].split(".")[0])
            block_numbers.append(block_num)
        except (ValueError, IndexError):
            continue

    if not block_numbers:
        return 1, 1

    last_block = max(block_numbers)
    start_from = (last_block * SAVE_EVERY) + 1
    next_block_id = last_block + 1
    print(f"🔄 Riprendo dal blocco {next_block_id}, combinazione {start_from}")
    return next_block_id, start_from

def run_backtests(df_train: pl.DataFrame, params_list: List[Dict[str, Any]],
                   timestamps: np.ndarray, rsi_np: np.ndarray,
                   bullish_np: np.ndarray, bearish_np: np.ndarray,
                   year: int, total_combinations: int
                   ) -> Tuple[float, Tuple, List]:
    best_result = None
    best_params = None
    best_orders = None
    buffer_orders = []
    counter = 0

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

        final_cash, orders = backtest_strategy(
            df_train, param, timestamps, rsi_np, bullish_np, bearish_np
        )
        buffer_orders.append((final_cash, orders))

        if counter % SAVE_EVERY == 0:
            save_results(buffer_orders, year, block_id)
            buffer_orders = []
            block_id += 1

        if best_result is None or final_cash > best_result:
            best_result = final_cash
            best_params = (
                param['sl'], param['tp'], param['rsi_entry'], 
                param['rsi_exit'], param['bb_std'], param['exposure']
            )
            best_orders = orders

    return best_result, best_params, best_orders

def main():
    sl_values = np.round(np.linspace(0.002, 0.010, 5), 4).tolist()
    tp_values = np.round(np.linspace(0.01, 0.03, 5), 4).tolist()
    rsi_entry_values = list(range(30, 46, 5))
    rsi_exit_values = list(range(55, 71, 5))
    bb_std_values = np.round(np.linspace(1.5, 2.5, 5), 2).tolist()
    exposures = np.round(np.linspace(0.1, 0.6, 6), 2).tolist()

    folder = './dati_forex/EURUSD/'
    df = load_forex_data(folder)

    years = list(range(2013, 2025))
    train_years_window = 1

    parameter_ranges = {
        "sl": sl_values,
        "tp": tp_values,
        "rsi_entry": rsi_entry_values,
        "rsi_exit": rsi_exit_values,
        "bb_std": bb_std_values,
        "exposure": exposures
    }

    all_params = generate_parameter_combinations(parameter_ranges)
    total_combinations = len(all_params)

    for year in years:
        print(f"\n🟩 Finestra: Train = {year}-{year + train_years_window - 1}, Test = {year + train_years_window}")

        df_train = df.filter(
            (pl.col("Datetime").dt.year() >= year) & 
            (pl.col("Datetime").dt.year() < year + train_years_window)
        )
        df_test = df.filter(pl.col("Datetime").dt.year() == year + train_years_window)

        if df_train.height == 0 or df_test.height == 0:
            print(f"⚠️ Dati mancanti per l'anno {year}, salto...")
            continue

        timestamps, rsi_np, bullish_np, bearish_np = calculate_indicators(df_train)

        best_result, best_params, best_orders = run_backtests(
            df_train, all_params, timestamps, rsi_np, bullish_np, bearish_np, year, total_combinations
        )

        save_results(None, year, 0, True, best_orders)
        print(f"✅ Train {year}: Miglior capitale = €{best_result:.2f} con parametri {best_params}")

if __name__ == "__main__":
    main()y'], 
                param['rsi_exit'], param['bb_std'], param['exposure']
            )
            best_orders = orders

    return best_result, best_params, best_orders

def main():
    sl_values = np.round(np.linspace(0.002, 0.010, 5), 4).tolist()
    tp_values = np.round(np.linspace(0.01, 0.03, 5), 4).tolist()
    rsi_entry_values = list(range(30, 46, 5))
    rsi_exit_values = list(range(55, 71, 5))
    bb_std_values = np.round(np.linspace(1.5, 2.5, 5), 2).tolist()
    exposures = np.round(np.linspace(0.1, 0.6, 6), 2).tolist()

    folder = './dati_forex/EURUSD/'
    df = load_forex_data(folder)

    years = list(range(2013, 2025))
    train_years_window = 1

    parameter_ranges = {
        "sl": sl_values,
        "tp": tp_values,
        "rsi_entry": rsi_entry_values,
        "rsi_exit": rsi_exit_values,
        "bb_std": bb_std_values,
        "exposure": exposures
    }

    all_params = generate_parameter_combinations(parameter_ranges)
    total_combinations = len(all_params)

    for year in years:
        print(f"\n🟩 Finestra: Train = {year}-{year + train_years_window - 1}, Test = {year + train_years_window}")

        df_train = df.filter(
            (pl.col("Datetime").dt.year() >= year) & 
            (pl.col("Datetime").dt.year() < year + train_years_window)
        )
        df_test = df.filter(pl.col("Datetime").dt.year() == year + train_years_window)

        if df_train.height == 0 or df_test.height == 0:
            print(f"⚠️ Dati mancanti per l'anno {year}, salto...")
            continue

        timestamps, rsi_np, bullish_np, bearish_np = calculate_indicators(df_train)

        best_result, best_params, best_orders = run_backtests(
            df_train, all_params, timestamps, rsi_np, bullish_np, bearish_np, year, total_combinations
        )

        save_results(None, year, 0, True, best_orders)
        print(f"✅ Train {year}: Miglior capitale = €{best_result:.2f} con parametri {best_params}")

if __name__ == "__main__":
    main()