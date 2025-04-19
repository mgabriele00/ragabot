import os
import pickle
import numpy as np
import polars as pl
import talib
from datetime import datetime

# === PARAMETRI COMBINAZIONE ===
PARAMS = {
    "sl": 0.006,
    "tp": 0.02,
    "rsi_entry": 35,
    "rsi_exit": 55,
    "bb_std": 1.75,
    "exposure": 0.6
}

DATA_FOLDER = "./dati_forex/EURUSD/"
ORDERS_FOLDER = "orders/final/"
FEATURES_OUTPUT = "features/orders_all_years_features.csv"
YEARS_INPUT = [2013, 2024]


def resolve_years(input):
    if isinstance(input, int):
        return [input]
    elif isinstance(input, list) and len(input) == 2 and all(isinstance(i, int) for i in input):
        return list(range(input[0], input[1] + 1))
    elif isinstance(input, list):
        return input
    else:
        raise ValueError("YEARS_INPUT deve essere int o lista")


def convert_to_polars_datetime(x):
    if isinstance(x, datetime):
        return pl.Series("", [x]).cast(pl.Datetime("ms"))[0]
    elif isinstance(x, np.datetime64):
        return pl.Series("", [x.item()]).cast(pl.Datetime("ms"))[0]
    elif isinstance(x, str):
        try:
            return pl.Series("", [datetime.strptime(x, "%Y-%m-%d %H:%M:%S")]).cast(pl.Datetime("ms"))[0]
        except ValueError:
            return pl.Series("", [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")]).cast(pl.Datetime("ms"))[0]
    else:
        raise TypeError(f"Formato datetime non supportato: {type(x)}")


def load_orders_from_final(year: int):
    path = os.path.join(ORDERS_FOLDER, f"orders_{year}_train.pkl")
    if not os.path.exists(path):
        print(f"⚠️ File mancante: {path}")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def load_forex_data(year):
    files = [f for f in sorted(os.listdir(DATA_FOLDER)) if str(year) in f and f.endswith(".csv")]
    dfs = []
    for file in files:
        df = pl.read_csv(os.path.join(DATA_FOLDER, file), has_header=False)
        df = df.select([
            pl.col("column_1").alias("Date"),
            pl.col("column_2").alias("Time"),
            pl.col("column_3").cast(pl.Float64).alias("Open"),
            pl.col("column_4").cast(pl.Float64).alias("High"),
            pl.col("column_5").cast(pl.Float64).alias("Low"),
            pl.col("column_6").cast(pl.Float64).alias("Close")
        ])
        df = df.with_columns([
            pl.concat_str(["Date", pl.lit(" "), "Time"]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
        ])
        dfs.append(df.select(["Datetime", "Open", "High", "Low", "Close"]))
    return pl.concat(dfs).sort("Datetime")


def calculate_indicators(df: pl.DataFrame, bb_std: float) -> pl.DataFrame:
    open_np = df["Open"].to_numpy()
    high_np = df["High"].to_numpy()
    low_np = df["Low"].to_numpy()
    close_np = df["Close"].to_numpy()

    rsi = talib.RSI(close_np, timeperiod=14)
    upper, middle, lower = talib.BBANDS(close_np, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)

    df = df.with_columns([
        pl.Series("RSI", rsi),
        pl.Series("BB_Upper", upper),
        pl.Series("BB_Middle", middle),
        pl.Series("BB_Lower", lower)
    ])

    for pattern in talib.get_function_groups()['Pattern Recognition']:
        func = getattr(talib, pattern)
        values = func(open_np, high_np, low_np, close_np)
        df = df.with_columns([pl.Series(pattern, values)])

    return df


def merge_orders_with_indicators(orders: list, df_indicators: pl.DataFrame, year: int) -> pl.DataFrame:
    df_ind = df_indicators.with_columns([
        pl.col("Datetime").cast(pl.Datetime("ms"))
    ])

    enriched_rows = []
    for order in orders:
        try:
            entry_time = convert_to_polars_datetime(order["Entry Time"])
        except Exception as e:
            print(f"⚠️ Errore conversione Entry Time: {order['Entry Time']} -> {e}")
            continue

        match = df_ind.filter(pl.col("Datetime") == entry_time)

        if match.is_empty():
            continue

        row = match.to_dicts()[0]
        row.update({
            "PnL": order["PnL"],
            "Success": int(order["PnL"] > 0),
            "Entry Time": entry_time,
            "Year": year
        })
        enriched_rows.append(row)

    return pl.DataFrame(enriched_rows)


# === MAIN ===
if __name__ == "__main__":
    all_dfs = []
    years = resolve_years(YEARS_INPUT)

    for year in years:
        print(f"\n\U0001F4E6 Anno {year} - Caricamento ordini e dati...")
        orders = load_orders_from_final(year)
        print(f"✅ Ordini trovati: {len(orders)}")
        if not orders:
            continue

        df = load_forex_data(year)
        print(f"✅ Dati caricati: {df.shape[0]} righe")
        print("⚙️ Calcolo indicatori RSI + Bollinger + Pattern...")
        df_ind = calculate_indicators(df, bb_std=PARAMS["bb_std"])

        print("🔗 Join tra ordini e indicatori...")
        df_feat = merge_orders_with_indicators(orders, df_ind, year)
        print(f"📊 Match riusciti: {df_feat.shape[0]} righe")

        if df_feat.shape[0] > 0:
            all_dfs.append(df_feat)

    if all_dfs:
        df_final = pl.concat(all_dfs)
        os.makedirs("features", exist_ok=True)
        df_final.write_csv(FEATURES_OUTPUT)
        print(f"\n💾 Salvato: {FEATURES_OUTPUT}")

        print("\n📈 Riepilogo per anno:")
        summary = df_final.groupby("Year").agg([
            pl.count().alias("Ordini"),
            pl.mean("PnL").round(4).alias("PnL Medio"),
            pl.mean("Success").round(4).alias("Success Rate")
        ]).sort("Year")
        print(summary)
    else:
        print("❌ Nessun dato disponibile.")