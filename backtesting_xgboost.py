import os
import pickle
import numpy as np
import pandas as pd
import polars as pl
import talib
import xgboost as xgb
from typing import Any, Dict, List, Tuple

# === PARAMETRI INIZIALI ===
FOLDER = './dati_forex/EURUSD/'
YEARS_INPUT = [2013, 2024]  # singolo anno o range o lista multipla
MERGE_YEARS = True
MODEL_PATH = "models/single_combination.model"
INITIAL_CASH = 1000
LEVERAGE = 100

# === PARAMETRI BASE ===
PARAMS = {
    "sl": 0.006,
    "tp": 0.02,
    "rsi_entry": 35,
    "rsi_exit": 55,
    "bb_std": 1.75,
    "exposure": 0.6
}

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

def calculate_indicators(df: pl.DataFrame, bb_std: float) -> Tuple[pd.DataFrame, np.ndarray]:
    df_pd = df.to_pandas()
    close = df_pd["Close"].values
    open_ = df_pd["Open"].values
    high = df_pd["High"].values
    low = df_pd["Low"].values
    time = df_pd["Datetime"].values

    df_pd["RSI"] = talib.RSI(close, timeperiod=14)
    upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)
    df_pd["BB_Upper"] = upper
    df_pd["BB_Middle"] = middle
    df_pd["BB_Lower"] = lower

    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        df_pd[pattern] = getattr(talib, pattern)(open_, high, low, close)

    return df_pd, time

def generate_signals(df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    return (
        (df["RSI"] < params["rsi_entry"]) &
        (df["Close"] < df["BB_Lower"]) &
        (df[[col for col in df.columns if col.startswith("CDL")]].max(axis=1) > 0)
    ).values

def backtest_con_xgboost(df: pd.DataFrame, time: np.ndarray, params: Dict[str, Any], model: Any) -> Tuple[float, List[Dict]]:
    entries = generate_signals(df, params)
    cash = INITIAL_CASH
    in_position = False
    orders = []

    for i in range(len(df)):
        price = df.iloc[i]["Close"]
        timestamp = time[i]

        if entries[i] and not in_position:
            features = df.iloc[[i]].drop(columns=["Datetime"])
            features = features[model.get_booster().feature_names]
            prob = model.predict_proba(features)[0, 1]
            if prob < 0.5:
                continue

            tp = params["tp"] * (5 if prob >= 0.65 else 1.0)
            sl = params["sl"] * (5 if prob >= 0.65 else 1.0)

            size_eur = cash * params["exposure"]
            size = (size_eur * LEVERAGE) / price

            entry_price = price
            entry_time = timestamp
            entry_size = size
            current_sl = sl
            current_tp = tp
            in_position = True

        elif in_position:
            sl_trigger = price <= entry_price * (1 - current_sl)
            tp_trigger = price >= entry_price * (1 + current_tp)
            exit_trigger = (df.iloc[i][[col for col in df.columns if col.startswith("CDL")]].min() < 0)
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
                    "Prob": round(prob, 4),
                    "Entry Time": entry_time,
                    "Exit Time": timestamp,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Size": entry_size,
                    "PnL": round(pnl, 6),
                    "Cash": cash,
                    "Reason": reason,
                    "SL": current_sl,
                    "TP": current_tp,
                    "Prob": round(prob, 4),
                    **params
                })
                in_position = False

    return cash, orders

def save_results(orders: List[Dict], name: str):
    os.makedirs("orders/final", exist_ok=True)
    with open(f"orders/final/orders_{name}_xgb.pkl", "wb") as f:
        pickle.dump(orders, f)
    pd.DataFrame(orders).to_csv(f"orders/final/orders_{name}_xgb.csv", index=False)

# === MAIN ===
if __name__ == "__main__":
    years = resolve_years(YEARS_INPUT)
    print(f"\n📅 Backtest con XGBoost per anni: {years} (MERGE={MERGE_YEARS})")

    print("📦 Caricamento modello XGBoost...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    if MERGE_YEARS:
        df = load_forex_data(FOLDER, years)
        df_indicators, time = calculate_indicators(df, PARAMS["bb_std"])
        final_cash, orders = backtest_con_xgboost(df_indicators, time, PARAMS, model)
        name = f"{years[0]}_{years[-1]}"
        print(f"✅ Capitale finale: €{final_cash:.2f} | Ordini: {len(orders)}")
        save_results(orders, name)
    else:
        for year in years:
            print(f"\n🔄 Backtest per anno: {year}")
            df = load_forex_data(FOLDER, [year])
            df_indicators, time = calculate_indicators(df, PARAMS["bb_std"])
            final_cash, orders = backtest_con_xgboost(df_indicators, time, PARAMS, model)
            print(f"✅ Capitale finale {year}: €{final_cash:.2f} | Ordini: {len(orders)}")
            save_results(orders, str(year))

    print("\n🏁 Backtest completato.")


import matplotlib.pyplot as plt

if orders:
    df_orders = pd.DataFrame(orders)
    df_orders["Entry Time"] = pd.to_datetime(df_orders["Entry Time"])
    df_orders["Exit Time"] = pd.to_datetime(df_orders["Exit Time"])

    # 1. Grafico capitale cumulato
    df_orders["Capital"] = df_orders["Cash"]
    plt.figure(figsize=(12, 6))
    plt.plot(df_orders["Exit Time"], df_orders["Capital"], label="Capitale")
    plt.xlabel("Tempo")
    plt.ylabel("Capitale (€)")
    plt.title("📈 Evoluzione del Capitale")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Grafico EURUSD con segnali di entrata/uscita
    print("📉 Generazione grafico prezzi + segnali...")
    df_plot = df[["Datetime", "Close"]].to_pandas()
    df_plot["Datetime"] = pd.to_datetime(df_plot["Datetime"])
    plt.figure(figsize=(14, 6))
    plt.plot(df_plot["Datetime"], df_plot["Close"], label="EUR/USD", alpha=0.5)

    # Entry points
    plt.scatter(df_orders["Entry Time"], df_orders["Entry Price"], marker="^", color="green", label="Entrata", s=40)
    # Exit points
    plt.scatter(df_orders["Exit Time"], df_orders["Exit Price"], marker="v", color="red", label="Uscita", s=40)

    plt.title("💱 EUR/USD con Entrate/Uscite")
    plt.xlabel("Tempo")
    plt.ylabel("Prezzo")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
