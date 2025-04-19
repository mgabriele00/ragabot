import os
import pickle
import numpy as np
import polars as pl
import talib
from typing import List, Dict

# === PARAMETRI FILTRO ===
PARAMS = {
    "sl": 0.006,
    "tp": 0.02,
    "rsi_entry": 35,
    "rsi_exit": 55,
    "bb_std": 1.75,
    "exposure": 0.6
}

# === Indicatori: RSI, Pattern, Bollinger Bands ===
def calculate_indicators(df: pl.DataFrame, bb_std: float = 1.75) -> pl.DataFrame:
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

# === Confronto robusto parametri ===
def _order_matches_params(order: dict, params: dict) -> bool:
    try:
        return (
            np.isclose(order['SL'], params['sl'], atol=1e-6) and
            np.isclose(order['TP'], params['tp'], atol=1e-6) and
            np.isclose(order['BB Std'], params['bb_std'], atol=1e-6) and
            np.isclose(order['Exposure'], params['exposure'], atol=1e-6) and
            order['RSI Entry'] == params['rsi_entry'] and
            order['RSI Exit'] == params['rsi_exit']
        )
    except KeyError as e:
        print(f"⚠️ Chiave mancante nell'ordine: {e}")
        return False

# === Caricamento ordini filtrati ===
def load_filtered_orders(folder: str, selected_params: dict) -> list:
    filtered_orders = []

    for file in sorted(os.listdir(folder)):
        if not file.endswith(".pkl") or not file.startswith("orders_"):
            continue

        path = os.path.join(folder, file)
        try:
            with open(path, "rb") as f:
                all_orders = pickle.load(f)
        except Exception as e:
            print(f"❌ Errore nel leggere {path}: {e}")
            continue

        if isinstance(all_orders, list):
            for item in all_orders:
                if isinstance(item, tuple):
                    _, order_list = item
                    for order in order_list:
                        if _order_matches_params(order, selected_params):
                            filtered_orders.append(order)
                elif isinstance(item, dict):
                    if _order_matches_params(item, selected_params):
                        filtered_orders.append(item)
        elif isinstance(all_orders, dict):
            if _order_matches_params(all_orders, selected_params):
                filtered_orders.append(all_orders)

    return filtered_orders

# === Join tra ordini e indicatori ===
def merge_orders_with_indicators(orders: list, df_ind: pl.DataFrame) -> pl.DataFrame:
    df_ind = df_ind.with_columns([
        pl.col("Datetime").cast(pl.Datetime("ms"))
    ])

    enriched_rows = []
    for order in orders:
        entry_time = order["Entry Time"]
        pnl = order["PnL"]

        row = df_ind.filter(pl.col("Datetime") == entry_time)
        if row.is_empty():
            print(f"❌ Entry time non trovato: {entry_time}")
            continue

        row_dict = row.to_dicts()[0]
        row_dict["PnL"] = pnl
        row_dict["Success"] = int(pnl > 0)
        row_dict["Entry Time"] = entry_time
        enriched_rows.append(row_dict)

    return pl.DataFrame(enriched_rows)

# === MAIN ===
if __name__ == "__main__":
    all_dfs = []

    for year in range(2013, 2025):
        print(f"\n📦 Elaborazione anno {year}...")

        # === Carica i CSV Forex ===
        files = sorted([f for f in os.listdir("./dati_forex/EURUSD/") if str(year) in f and f.endswith(".csv")])
        df_list = []
        for f in files:
            temp_df = pl.read_csv(f"./dati_forex/EURUSD/{f}", has_header=False)
            temp_df = temp_df.select([
                pl.col("column_1").alias("Date"),
                pl.col("column_2").alias("Time"),
                pl.col("column_3").cast(pl.Float64).alias("Open"),
                pl.col("column_4").cast(pl.Float64).alias("High"),
                pl.col("column_5").cast(pl.Float64).alias("Low"),
                pl.col("column_6").cast(pl.Float64).alias("Close")
            ]).with_columns([
                pl.concat_str([
                    pl.col("Date"),
                    pl.lit(" "),
                    pl.col("Time")
                ]).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M").alias("Datetime")
            ]).select(["Datetime", "Open", "High", "Low", "Close"])
            df_list.append(temp_df)

        if not df_list:
            continue

        df = pl.concat(df_list).sort("Datetime")
        print(f"✅ Dati caricati: {df.shape[0]} righe")

        # === Calcolo indicatori ===
        print("⚙️ Calcolo indicatori (RSI, BBANDS, Pattern)...")
        df_indicators = calculate_indicators(df, bb_std=PARAMS["bb_std"])

        # === Caricamento ordini ===
        print("📦 Caricamento ordini...")
        orders = load_filtered_orders(f"orders/partial/{year}", PARAMS)
        print(f"✅ Ordini trovati: {len(orders)}")

        # === Join indicatori + ordini
        print("🔗 Join con indicatori...")
        df_features = merge_orders_with_indicators(orders, df_indicators)

        # === Riepilogo
        num_orders = len(orders)
        num_merged = df_features.shape[0]
        print(f"📊 Ordini = {num_orders} | Match con indicatori = {num_merged}")

        # === Aggiungi colonna Year
        if num_merged > 0:
            df_features = df_features.with_columns([
                pl.lit(year).alias("Year")
            ])
            all_dfs.append(df_features)

    # === Salvataggio e riepilogo finale
    if all_dfs:
        df_final = pl.concat(all_dfs)
        print(f"\n✅ Dataset finale: {df_final.shape}")
        os.makedirs("features", exist_ok=True)
        df_final.write_csv("features/orders_all_years_features.csv")
        print("💾 Salvato in: features/orders_all_years_features.csv")

        # Riepilogo
        print("\n📈 Riepilogo per anno:")
        summary = df_final.groupby("Year").agg([
            pl.count().alias("Ordini"),
            pl.mean("PnL").round(4).alias("PnL Medio"),
            pl.mean("Success").round(4).alias("Success Rate")
        ]).sort("Year")
        print(summary)
    else:
        print("❌ Nessun dato generato.")
