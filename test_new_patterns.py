import os
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from patternpy.pattern_detector import PatternDetector

# === PARAMETRI ===
FOLDER = './dati_forex/EURUSD/'
YEARS = [2022]  # puoi mettere [2022, 2023] o un range

# === Caricamento dati ===
def load_forex_data(folder: str, years: list) -> pd.DataFrame:
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
        dfs.append(df.select(["Datetime", "Close"]))
    df_all = pl.concat(dfs).sort("Datetime")
    return df_all.to_pandas()

# === MAIN ===
if __name__ == "__main__":
    print("📥 Caricamento dati EUR/USD...")
    df = load_forex_data(FOLDER, YEARS)
    close_prices = df["Close"].values
    timestamps = df["Datetime"].values

    print("🔍 Rilevamento Testa e Spalle con PatternPy...")
    detector = PatternDetector()
    patterns = detector.detect_head_and_shoulders(close_prices)

    if not patterns:
        print("❌ Nessun pattern trovato.")
    else:
        print(f"✅ Trovati {len(patterns)} pattern. Visualizzazione...")

        for i, pattern in enumerate(patterns):
            start_idx = pattern['start_index']
            end_idx = pattern['end_index']
            segment = close_prices[start_idx:end_idx + 1]
            segment_time = timestamps[start_idx:end_idx + 1]

            plt.figure(figsize=(12, 4))
            plt.plot(segment_time, segment, label="Prezzo")
            plt.title(f"🟢 Testa e Spalle #{i+1} ({segment_time[0]} → {segment_time[-1]})")
            plt.xlabel("Tempo")
            plt.ylabel("Prezzo")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
