import os
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import talib
import numpy as np
import datetime as dt

# Imposta backend interattivo
matplotlib.use("TkAgg")  # "TkAgg" o "Qt5Agg" per macOS/Linux/Windows
plt.ion()

# === PARAMETRI ===
FOLDER = "./dati_forex/EURUSD/"
YEAR = 2013
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Caricamento dati ===
print(f"📦 Caricamento dati per l'anno {YEAR}...")
all_files = sorted([f for f in os.listdir(FOLDER) if f.endswith(".csv") and str(YEAR) in f])

# Unione dati
df_list = []
for file in all_files:
    df = pl.read_csv(os.path.join(FOLDER, file), has_header=False)
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
    df_list.append(df.select(["Datetime", "Open", "High", "Low", "Close"]))

df_all = pl.concat(df_list).sort("Datetime")
print(f"✅ Dati caricati: {df_all.shape[0]} righe")

# === Filtro per intervallo ristretto ===
df_filtered = df_all.filter(
    (pl.col("Datetime") >= dt.datetime(YEAR, 1, 2, 0, 0)) &
    (pl.col("Datetime") <= dt.datetime(YEAR, 1, 5, 23, 59))
)

# === Indicatori ===
open_ = df_filtered["Open"].to_numpy()
high = df_filtered["High"].to_numpy()
low = df_filtered["Low"].to_numpy()
close = df_filtered["Close"].to_numpy()
dates = df_filtered["Datetime"].to_numpy()

rsi = talib.RSI(close, timeperiod=14)
upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=1.75, nbdevdn=1.75)

bullish = np.zeros(len(close), dtype=bool)
pattern_map = {}
pattern_names = talib.get_function_groups()['Pattern Recognition']

for name in pattern_names:
    func = getattr(talib, name)
    result = func(open_, high, low, close)
    is_pattern = result != 0
    bullish = bullish | (result > 0)
    pattern_map[name] = is_pattern

entries = (rsi < 35) & (close < lower) & bullish

# === Preparo dati per candlestick ===
ohlc_data = [
    [mdates.date2num(dates[i]), open_[i], high[i], low[i], close[i]]
    for i in range(len(dates))
]

# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

candlestick_ohlc(ax1, ohlc_data, width=1/1440, colorup='green', colordown='red')
ax1.plot(dates, upper, linestyle="--", label="Bollinger Upper")
ax1.plot(dates, lower, linestyle="--", label="Bollinger Lower")
ax1.scatter(dates[entries], close[entries], color='blue', marker='^', label='Signal Entry', s=40)
ax1.set_title("EUR/USD Candlestick con Segnali e Bande di Bollinger")
ax1.legend()
ax1.grid(True)

ax2.plot(dates, rsi, label="RSI(14)", color="blue")
ax2.axhline(70, color="red", linestyle="--")
ax2.axhline(30, color="green", linestyle="--")
ax2.set_title("RSI")
ax2.set_ylabel("RSI")
ax2.grid(True)

plt.tight_layout()
plt.show(block=True)  # Blocca l'esecuzione finché non chiudi la finestra

# Salva grafico dopo che è stato mostrato
fig.savefig(f"{OUTPUT_DIR}/candlestick_signals_zoom.png")

# === Stampa pattern attivi sugli entry ===
print("\n📌 Pattern attivi durante le Entry:")
for i in np.where(entries)[0]:
    active = [name for name, mask in pattern_map.items() if mask[i]]
    print(f"{dates[i]}: {', '.join(active)}")
