import os
import polars as pl
import numpy as np
import talib
import itertools
import pandas as pd

INITIAL_CASH = 1000
LEVERAGE = 100
SAVE_EVERY = 20000
FOLDER = './dati_forex/EURUSD/'

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

def save_results(orders, year, idx):
    folder = f'orders/sim_short/{year}'
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame(orders).to_csv(f'{folder}/orders_{year}_part_{idx}.csv', index=False)
    print(f"💾 CSV salvato: {folder}/orders_{year}_part_{idx}.csv")

if __name__ == '__main__':
    combinations = generate_combinations(PARAM_GRID)

    for year in YEARS_INPUT:
        df = load_forex_data(year)
        indicators = calculate_indicators(df)
        all_orders = []

        for idx, params in enumerate(combinations, 1):
            print(f"🔄 Anno {year} | Combinazione {idx}/{len(combinations)}")
            final_cash, orders = backtest(df, indicators, params, idx, year)
            all_orders.extend(orders)

            if idx % SAVE_EVERY == 0:
                save_results(all_orders, year, idx)
                all_orders = []

        if all_orders:
            save_results(all_orders, year, 'final')
