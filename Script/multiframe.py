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
USE_STOCHASTIC_RSI = False

# === Parametri Multi-Frame ===
HIGHER_TIMEFRAME = "15m"  # 👈 Scegli il timeframe superiore ("5m", "15m", ecc.)
USE_SUPERTREND = True
USE_EMA = False
EMA_PERIOD = 50
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

PARAMS = {
    "rsi_entry": 35,
    "rsi_exit": 55,
    "bb_std": 1.75,
    "exposure": 0.6,
    "atr_window": 20,
    "atr_factor": 1.5
}
INITIAL_CASH = 1000
LEVERAGE = 100

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

def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = uptrend, -1 = downtrend
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        supertrend[i] = lowerband[i] if direction[i] == 1 else upperband[i]
    return supertrend, direction

def create_higher_timeframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    df_high = df.group_by_dynamic("Datetime", every=timeframe, closed="right").agg([
        pl.col("Open").first().alias("Open"),
        pl.col("High").max().alias("High"),
        pl.col("Low").min().alias("Low"),
        pl.col("Close").last().alias("Close")
    ]).drop_nulls()
    return df_high


def calculate_indicators(df: pl.DataFrame, use_stochastic_rsi: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = df["Close"].to_numpy()
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    time = df["Datetime"].to_numpy()

    if use_stochastic_rsi:
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        rsi = fastk
    else:
        rsi = talib.RSI(close, timeperiod=14)

    bullish = np.zeros(len(close), dtype=bool)
    bearish = np.zeros(len(close), dtype=bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        values = getattr(talib, pattern)(open_, high, low, close)
        bullish |= values > 0
        bearish |= values < 0

    return time, rsi, bullish, bearish

def prepare_multiframe_features(df_m1: pl.DataFrame) -> pl.DataFrame:
    df_m5 = create_higher_timeframe(df_m1, timeframe=HIGHER_TIMEFRAME)
    close_m5 = df_m5["Close"].to_numpy()
    high_m5 = df_m5["High"].to_numpy()
    low_m5 = df_m5["Low"].to_numpy()

    if USE_EMA:
        ema = talib.EMA(close_m5, timeperiod=EMA_PERIOD)
        df_m5 = df_m5.with_columns([pl.Series(name="EMA", values=ema)])

    if USE_SUPERTREND:
        _, supertrend_dir = calculate_supertrend(high_m5, low_m5, close_m5, atr_period=SUPERTREND_ATR_PERIOD, multiplier=SUPERTREND_MULTIPLIER)
        df_m5 = df_m5.with_columns([pl.Series(name="Supertrend_Dir", values=supertrend_dir)])

    # Join dinamico su 1 minuto
    join_cols = ["Datetime"]
    join_df = ["EMA", "Supertrend_Dir"]
    df_join = df_m5.select(["Datetime"] + [col for col in join_df if col in df_m5.columns])

    df_final = df_m1.join_asof(df_join, on="Datetime", strategy="backward")

    return df_final

def generate_signals(df: pl.DataFrame, close: np.ndarray, rsi: np.ndarray, bullish: np.ndarray, bearish: np.ndarray, params: Dict[str, Any]):
    bb_std = params["bb_std"]
    rsi_entry = params["rsi_entry"]
    rsi_exit = params["rsi_exit"]

    upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)
    bollinger_width = (upper - lower) / middle
    width_threshold = 0.001

    bands_are_narrow = bollinger_width < width_threshold

    # Conversione di close in Series Polars
    close_series = pl.Series(name="close_tmp", values=close)

    ema_condition_long = (df["EMA"] < close_series) if "EMA" in df.columns else True
    ema_condition_short = (df["EMA"] > close_series) if "EMA" in df.columns else True

    supertrend_condition_long = (df["Supertrend_Dir"] == 1) if "Supertrend_Dir" in df.columns else True
    supertrend_condition_short = (df["Supertrend_Dir"] == -1) if "Supertrend_Dir" in df.columns else True

    long_filter = ema_condition_long & supertrend_condition_long
    short_filter = ema_condition_short & supertrend_condition_short

    entries_long = (rsi < rsi_entry) & (close < lower) & bullish & bands_are_narrow & long_filter.to_numpy()
    exits_long = (rsi > rsi_exit) & (close > upper) & bearish

    entries_short = (rsi > rsi_exit) & (close > upper) & bearish & bands_are_narrow & short_filter.to_numpy()
    exits_short = (rsi < rsi_entry) & (close < lower) & bullish

    return entries_long, exits_long, entries_short, exits_short



def backtest(df: pl.DataFrame, time: np.ndarray, rsi: np.ndarray, bullish: np.ndarray, bearish: np.ndarray, params: Dict[str, Any]) -> Tuple[float, List[Dict]]:
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    atr = talib.ATR(high, low, close, timeperiod=params["atr_window"])

    entries_long, exits_long, entries_short, exits_short = generate_signals(df, close, rsi, bullish, bearish, params)
    cash = INITIAL_CASH
    in_position = False
    is_short = False
    orders = []

    for i in range(len(close)):
        price = close[i]
        timestamp = time[i]

        if not in_position:
            if entries_long[i]:
                direction = "LONG"
                entry_price = price
                entry_time = timestamp
                is_short = False
            elif entries_short[i]:
                direction = "SHORT"
                entry_price = price
                entry_time = timestamp
                is_short = True
            else:
                continue

            size_eur = cash * params["exposure"]
            size = (size_eur * LEVERAGE) / price
            entry_size = size
            atr_val = atr[i] if not np.isnan(atr[i]) else 0
            sl_val = atr_val * params["atr_factor"]
            tp_val = atr_val * params["atr_factor"] * 2
            sl_price = entry_price - sl_val if not is_short else entry_price + sl_val
            tp_price = entry_price + tp_val if not is_short else entry_price - tp_val
            in_position = True

        elif in_position:
            if is_short:
                sl_trigger = price >= sl_price
                tp_trigger = price <= tp_price
                exit_trigger = exits_short[i]
            else:
                sl_trigger = price <= sl_price
                tp_trigger = price >= tp_price
                exit_trigger = exits_long[i]

            reason = None
            if sl_trigger:
                reason = "Stop Loss"
            elif tp_trigger:
                reason = "Take Profit"
            elif exit_trigger:
                reason = "Signal Exit"

            if reason:
                exit_price = price
                pnl = (entry_price - exit_price) * entry_size if is_short else (exit_price - entry_price) * entry_size
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
                    "Type": "SHORT" if is_short else "LONG",
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
        df_m1 = load_forex_data(FOLDER, years)
        print(f"\n🟦 Backtest MERGE su anni: {years}")
        df_m1 = prepare_multiframe_features(df_m1)
        time, rsi, bull, bear = calculate_indicators(df_m1, use_stochastic_rsi=USE_STOCHASTIC_RSI)
        final_cash, orders = backtest(df_m1, time, rsi, bull, bear, PARAMS)
        name = f"{years[0]}_{years[-1]}"
        print(f"✅ Capitale finale: €{final_cash:.2f}")
        save_results(orders, name)
        order_counts[name] = len(orders)
    else:
        for year in years:
            print(f"\n🟦 Backtest per anno: {year}")
            df_m1 = load_forex_data(FOLDER, [year])
            df_m1 = prepare_multiframe_features(df_m1)
            time, rsi, bull, bear = calculate_indicators(df_m1, use_stochastic_rsi=USE_STOCHASTIC_RSI)
            final_cash, orders = backtest(df_m1, time, rsi, bull, bear, PARAMS)
            print(f"✅ Capitale finale {year}: €{final_cash:.2f}")
            save_results(orders, str(year))
            order_counts[str(year)] = len(orders)

    print("\n📊 Conteggio ordini per anno:")
    for y, count in order_counts.items():
        print(f"🗓️ {y}: {count} ordini")

    print("\n🏆 Conteggio ordini VINCOLI per anno (PnL > 0):")
    for year in order_counts:
     pkl_path = f"orders/final/orders_{year}_train.pkl"
     if not os.path.exists(pkl_path):
         continue

     with open(pkl_path, "rb") as f:
         orders = pickle.load(f)

     if not orders:
         continue

     df = pd.DataFrame(orders)
     wins = (df["PnL"] > 0).sum()
     total = len(df)
     winrate = wins / total * 100 if total > 0 else 0

     print(f"🗓️ {year}: {wins}/{total} ordini vinti ({winrate:.2f}% win rate)")

