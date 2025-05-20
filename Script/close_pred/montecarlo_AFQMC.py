import polars as pl
import numpy as np
from ta.volatility import AverageTrueRange
from numba import njit
from sklearn.metrics import classification_report

# === Parametri ===
tp_mult = 0.1
sl_mult = 20
threshold = 0.7
n_sim = 10000
lookahead = 10
window_sigma = 100

# === 1. Caricamento CSV ===
df_raw = pl.read_csv("Script/close_pred/data/EURUSD_M1_2013_2024.csv")
df_raw = df_raw.with_columns([
    pl.concat_str([pl.col("Date"), pl.col("Time")], separator=" ").str.to_datetime().alias("datetime")
]).drop(["Date", "Time"]).sort("datetime")

# Rinomina per compatibilità
df_raw = df_raw.rename({
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# === 2. Calcolo ATR in Pandas ===
df_pd = df_raw.to_pandas()
atr = AverageTrueRange(high=df_pd["high"], low=df_pd["low"], close=df_pd["close"], window=14).average_true_range()
df_pd["atr"] = atr
df_pd["log_ret"] = np.log(df_pd["close"] / df_pd["close"].shift(1))
df_pd["sigma"] = df_pd["log_ret"].rolling(window_sigma).std()
df_pd["tp"] = df_pd["close"] + df_pd["atr"] * tp_mult
df_pd["sl"] = df_pd["close"] - df_pd["atr"] * sl_mult

df = pl.from_pandas(df_pd).drop_nulls()

# === 3. Simulatore Monte Carlo con Importance Sampling (AFQMC-like) ===
@njit(fastmath=True)
def simulate_close_afqmc(current_price, sigma, n_sim, dt=1/1440):
    shift = 0.1  # Campo ausiliario (bias) arbitrario, da calibrare
    Z = np.random.randn(n_sim) + shift
    weights = np.exp(-0.5 * ((Z - shift)**2 - Z**2))
    log_ret = (-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = current_price * np.exp(log_ret)
    return prices, weights

# === 4. Loop con simulazione ponderata ===
exploded_rows = []

df_len = df.height
close = df["close"].to_numpy()
high = df["high"].to_numpy()
low = df["low"].to_numpy()
tp_all = df["tp"].to_numpy()
sl_all = df["sl"].to_numpy()
sigma_all = df["sigma"].to_numpy()
datetime = df["datetime"].to_numpy()

for t in range(df_len - lookahead - 2):
    tp = tp_all[t]
    sl = sl_all[t]
    if np.isnan(tp) or np.isnan(sl):
        continue

    for offset in range(1, lookahead + 1):
        i = t + offset
        if i + 1 >= df_len:
            break

        price_now = close[i]
        sigma = sigma_all[i]
        if np.isnan(price_now) or np.isnan(sigma):
            continue

        sims, weights = simulate_close_afqmc(price_now, sigma, n_sim, dt=1)

        mask_tp = sims >= tp
        mask_sl = sims <= sl

        prob_tp = np.sum(weights[mask_tp]) / np.sum(weights)
        prob_sl = np.sum(weights[mask_sl]) / np.sum(weights)

        if prob_tp >= threshold:
            pred = 1
        elif prob_sl >= threshold:
            pred = -1
        else:
            pred = 0

        h_next = high[i + 1]
        l_next = low[i + 1]

        if h_next >= tp:
            outcome_real = 1
        elif l_next <= sl:
            outcome_real = -1
        else:
            outcome_real = 0

        correct = pred == outcome_real

        exploded_rows.append({
            "base_index": datetime[t],
            "future_index": datetime[i],
            "offset": offset,
            "tp": tp,
            "sl": sl,
            "future_close": price_now,
            "prob_tp": prob_tp,
            "prob_sl": prob_sl,
            "pred": pred,
            "outcome_real": outcome_real,
            "correct": correct
        })

# === 5. Costruzione DataFrame e metriche ===
df_exploded = pl.from_dicts(exploded_rows)

y_true = df_exploded["outcome_real"].to_numpy()
y_pred = df_exploded["pred"].to_numpy()

print("\n🎯 CLASSIFICATION REPORT (simulazioni AFQMC-like):\n")
print(classification_report(y_true, y_pred, labels=[1, 0, -1]))

accuracy = np.mean(y_true == y_pred)
coverage = np.mean(y_pred != 0)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Coverage (pred ≠ 0): {coverage:.4f}")

# === 6. Salvataggio finale ===
df_exploded.write_parquet("montecarlo_afqmc_polars.parquet")
