import os
import pickle
import numpy as np
import pandas as pd
import polars as pl
import talib
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

# === PARAMETRI INIZIALI ===
FOLDER = './dati_forex/EURUSD/'
YEARS_INPUT = [2013]
MERGE_YEARS = False
PARAMS = {
    "sl": 0.006,
    "tp": 0.02,
    "rsi_entry": 35,
    "rsi_exit": 55,
    "bb_std": 1.75,
    "exposure": 0.6
}
INITIAL_CASH = 1000
LEVERAGE = 100

# === UTILITY ===
def resolve_years(input: Any) -> List[int]:
    if isinstance(input, int):
        return [input]
    elif isinstance(input, list) and len(input) == 2:
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

# === INDICATORI ===
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

def generate_signals(close: np.ndarray, rsi: np.ndarray, bullish: np.ndarray,
                     params: Dict[str, Any], high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bb_std = params["bb_std"]
    upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)
    macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    entries = (rsi < params["rsi_entry"]) & (close < lower) & bullish
    exits = macd < macdsignal

    return entries, exits

# === BACKTEST ===
def backtest(df: pl.DataFrame, time: np.ndarray, rsi: np.ndarray,
             bullish: np.ndarray, bearish: np.ndarray, params: Dict[str, Any]) -> Tuple[float, List[Dict]]:
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    entries, exits = generate_signals(close, rsi, bullish, params, high, low)

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
            in_position = True

        elif in_position:
            sl_trigger = price <= entry_price * (1 - params["sl"])
            tp_trigger = price >= entry_price * (1 + params["tp"])
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

# === SALVATAGGIO ===
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

    os.makedirs("features", exist_ok=True)

    for year in order_counts:
        pkl_path = f"orders/final/orders_{year}_train.pkl"
        if not os.path.exists(pkl_path):
            continue

        print(f"\n📉 Grafico Entry/Exit + Ichimoku - Anno {year}")
        with open(pkl_path, "rb") as f:
            orders = pickle.load(f)

        if not orders:
            continue

        df_orders = pd.DataFrame(orders)
        df_orders["Entry Time"] = pd.to_datetime(df_orders["Entry Time"])
        df_orders["Exit Time"] = pd.to_datetime(df_orders["Exit Time"])

        anno = int(str(year).split("_")[0])
        df_price = load_forex_data(FOLDER, [anno]).to_pandas()
        df_price["Datetime"] = pd.to_datetime(df_price["Datetime"])

        span_a, span_b = calculate_ichimoku(
            df_price["High"].values,
            df_price["Low"].values,
            df_price["Close"].values
        )

        plt.figure(figsize=(14, 6))
        plt.plot(df_price["Datetime"], df_price["Close"], label="EUR/USD Close", color="gray", linewidth=1)
        plt.fill_between(df_price["Datetime"], span_a, span_b, where=span_a >= span_b, color="green", alpha=0.2, label="Kumo Bullish")
        plt.fill_between(df_price["Datetime"], span_a, span_b, where=span_a < span_b, color="red", alpha=0.2, label="Kumo Bearish")

        plt.scatter(df_orders["Entry Time"], df_orders["Entry Price"], color="green", marker="^", label="Entry", s=30)
        plt.scatter(df_orders["Exit Time"], df_orders["Exit Price"], color="red", marker="v", label="Exit", s=30)

        plt.title(f"📍 Entry & Exit con Ichimoku - Anno {year}")
        plt.xlabel("Data")
        plt.ylabel("Prezzo EUR/USD")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"features/entry_exit_ichimoku_{year}.png")
        plt.show()
