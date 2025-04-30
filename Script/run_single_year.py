import os
import pickle
import numpy as np
import pandas as pd
import polars as pl
import talib
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

# === PARAMETRI INIZIALI ===
FOLDER = '../dati_forex/EURUSD/'
YEARS_INPUT = [2013]
MERGE_YEARS = False
PARAMS = {
    "rsi_entry": 30,
    "rsi_exit": 55,
    "bb_std": 1.5,
    "exposure": 0.3,
    "acr_window": 14,
    "acr_factor": 1
}
INITIAL_CASH = 1000
LEVERAGE = 30

# === FUNZIONI DI UTILITÀ ===
def resolve_years(input: Any) -> List[int]:
    if isinstance(input, int):
        return [input]
    elif isinstance(input, list) and len(input) == 2 and all(isinstance(i, int) for i in input):
        return list(range(input[0], input[1] + 1))
    elif isinstance(input, list):
        return input
    else:
        raise ValueError("YEARS_INPUT deve essere int o lista")

def load_forex_data(folder: str, years: List[int]) -> pl.DataFrame:
    files = [f for f in sorted(os.listdir(folder)) if f.endswith(".csv") and any(str(y) in f for y in years)]
    dfs = []
    for file in files:
        df = pl.read_csv(os.path.join(folder, file), has_header=False)
        df = df.select([
            pl.col("column_1").alias("Date"),
            pl.col("column_2").alias("Time"),
            pl.col("column_3").cast(pl.Float64).alias("Open"),
            pl.col("column_4").cast(pl.Float64).alias("High"),
            pl.col("column_5").cast(pl.Float64).alias("Low"),
            pl.col("column_6").cast(pl.Float64).alias("Close"),
        ])
        df = df.with_columns([
            pl.concat_str(["Date", pl.lit(" "), "Time"]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
        ])
        dfs.append(df.select(["Datetime", "Open", "High", "Low", "Close"]))
    return pl.concat(dfs).sort("Datetime")

def calculate_indicators(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = df["Close"].to_numpy()
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    time = df["Datetime"].to_numpy()

    rsi = talib.RSI(close, timeperiod=14)
    bullish = np.zeros(len(close), dtype=bool)
    bearish = np.zeros(len(close), dtype=bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        values = getattr(talib, pattern)(open_, high, low, close)
        bullish |= values > 0
        bearish |= values < 0
    return time, rsi, bullish, bearish

def generate_signals(close: np.ndarray, rsi: np.ndarray, bullish: np.ndarray, bearish: np.ndarray, params: Dict[str, Any]):
    bb_std = params["bb_std"]
    rsi_entry = params["rsi_entry"]
    rsi_exit = params["rsi_exit"]
    upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)

    entries = (rsi < rsi_entry) & (close < lower) & bullish
    exits = (rsi > rsi_exit) & (close > upper) & bearish

    return entries, exits

def compute_acr(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(high - low).rolling(window).mean().to_numpy()

def backtest(df: pl.DataFrame, time: np.ndarray, rsi: np.ndarray, bullish: np.ndarray, bearish: np.ndarray, params: Dict[str, Any]) -> Tuple[float, List[Dict]]:
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    acr = compute_acr(high, low, params["acr_window"])

    entries, exits = generate_signals(close, rsi, bullish, bearish, params)
    cash = INITIAL_CASH
    in_position = False
    orders = []

    for i in range(len(close)):
        price = close[i]
        timestamp = time[i]

        if entries[i] and not in_position:
            size_eur = cash * params["exposure"]
            size = (size_eur * LEVERAGE) / price
            entry_price = price
            entry_time = timestamp
            entry_size = size
            acr_val = acr[i] if not np.isnan(acr[i]) else 0
            sl_val = acr_val * params["acr_factor"]
            tp_val = acr_val * params["acr_factor"] * 2
            sl_price = entry_price - sl_val
            tp_price = entry_price + tp_val
            in_position = True

        elif in_position:
            sl_trigger = price <= sl_price
            tp_trigger = price >= tp_price
            exit_trigger = exits[i]

            reason = None
            if sl_trigger:
                reason = "Stop Loss"
            elif tp_trigger:
                reason = "Take Profit"
            elif exit_trigger:
                reason = "Signal Exit"

            if reason:
                exit_price = price
                pnl = (exit_price - entry_price) * entry_size
                cash += pnl
                orders.append({
                    "Entry Time": entry_time,
                    "Exit Time": timestamp,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Size": entry_size,
                    "PnL": pnl,
                    "Cash": cash,
                    "Reason": reason,
                    **params
                })
                in_position = False

    return cash, orders

def save_results(orders: List[Dict], name: str):
    os.makedirs("orders/final", exist_ok=True)
    with open(f"orders/final/orders_{name}_train.pkl", "wb") as f:
        pickle.dump(orders, f)

    os.makedirs(f"orders/csv/{name}", exist_ok=True)
    pd.DataFrame(orders).to_csv(f"orders/csv/{name}/orders_{name}.csv", index=False)

# === MAIN ===
if __name__ == "__main__":
    years = resolve_years(YEARS_INPUT)
    order_counts = {}

    if MERGE_YEARS:
        df = load_forex_data(FOLDER, years)
        print(f"\n🟦 Backtest MERGE su anni: {years}")
        time, rsi, bull, bear = calculate_indicators(df)
        final_cash, orders = backtest(df, time, rsi, bull, bear, PARAMS)
        name = f"{years[0]}_{years[-1]}"
        print(f"✅ Capitale finale: €{final_cash:.2f}")
        save_results(orders, name)
        order_counts[name] = len(orders)
    else:
        for year in years:
            print(f"\n🟦 Backtest per anno: {year}")
            df = load_forex_data(FOLDER, [year])
            time, rsi, bull, bear = calculate_indicators(df)
            final_cash, orders = backtest(df, time, rsi, bull, bear, PARAMS)
            print(f"✅ Capitale finale {year}: €{final_cash:.2f}")
            save_results(orders, str(year))
            order_counts[str(year)] = len(orders)

    print("\n📊 Conteggio ordini per anno:")
    for y, count in order_counts.items():
        print(f"🗓️ {y}: {count} ordini")

    # === GRAFICI ===
    os.makedirs("features", exist_ok=True)

    for year in order_counts:
        pkl_path = f"orders/final/orders_{year}_train.pkl"
        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            orders = pickle.load(f)

        if not orders:
            continue

        df = pd.DataFrame(orders)
        df["Exit Time"] = pd.to_datetime(df["Exit Time"])

        plt.figure(figsize=(12, 5))
        plt.plot(df["Exit Time"], df["Cash"], label=f"Capitale ({year})")
        plt.title(f"📈 Andamento del Capitale - Anno {year}")
        plt.xlabel("Data")
        plt.ylabel("Capitale (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"features/curva_capitale_{year}.png")
        plt.show()

        plt.figure(figsize=(14, 6))
        df["Entry Time"] = pd.to_datetime(df["Entry Time"])
        plt.plot(df["Entry Time"], df["Entry Price"], label="Entry", color="green", marker="^", linestyle="None")
        plt.plot(df["Exit Time"], df["Exit Price"], label="Exit", color="red", marker="v", linestyle="None")
        plt.title(f"📍 Entry & Exit - Anno {year}")
        plt.xlabel("Data")
        plt.ylabel("Prezzo EUR/USD")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"features/entry_exit_aligned_{year}.png")
        plt.show()