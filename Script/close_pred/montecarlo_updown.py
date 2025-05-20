import polars as pl
import numpy as np
from ta.volatility import AverageTrueRange
from numba import njit
from sklearn.metrics import classification_report

# === Parametri ===
tp_mult          = 0.1
sl_mult          = 20
threshold        = 0.4    # <-- imposta qui la soglia che vuoi usare
n_sim            = 10000
lookahead        = 10
window_sigma     = 100
test_sample_size = 1000     # metti un intero per limitare il numero di barre, o None per usarle tutte

# === 1. Caricamento CSV e parsing datetime ===
df_raw = pl.read_csv("Script/close_pred/data/EURUSD_M1_2013_2024.csv")
df_raw = df_raw.with_columns(
    (pl.col("Date") + pl.lit(" ") + pl.col("Time"))
    .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M", strict=False)
    .alias("datetime")
).drop(["Date","Time"]).sort("datetime")

df_raw = df_raw.rename({
    "Open":   "open",
    "High":   "high",
    "Low":    "low",
    "Close":  "close",
    "Volume": "volume"
})

# === 2. ATR, log-ret, sigma, TP/SL ===
df_pd = df_raw.to_pandas()
atr = AverageTrueRange(
    high=df_pd["high"], low=df_pd["low"], close=df_pd["close"], window=14
).average_true_range()

df_pd["atr"]     = atr
df_pd["log_ret"] = np.log(df_pd["close"] / df_pd["close"].shift(1))
df_pd["sigma"]   = df_pd["log_ret"].rolling(window_sigma).std()
df_pd["tp"]      = df_pd["close"] + df_pd["atr"] * tp_mult
df_pd["sl"]      = df_pd["close"] - df_pd["atr"] * sl_mult

df = pl.from_pandas(df_pd).drop_nulls()

# === 3. Simulatore Monte Carlo ===
@njit(fastmath=True)
def simulate_close_numba(current_price, sigma, n_sim, dt=1.0, mu=0.0):
    Z = np.random.randn(n_sim)
    log_ret = (mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt)*Z
    return current_price * np.exp(log_ret)

# === 4. Loop esploso e simulazioni ===
exploded = []
close_arr = df["close"].to_numpy()
high_arr  = df["high"].to_numpy()
low_arr   = df["low"].to_numpy()
tp_arr    = df["tp"].to_numpy()
sl_arr    = df["sl"].to_numpy()
sigma_arr = df["sigma"].to_numpy()

N = df.height
if test_sample_size is not None and test_sample_size < N:
    N = test_sample_size

for t in range(N - lookahead - 1):
    base_tp = tp_arr[t]
    base_sl = sl_arr[t]
    if np.isnan(base_tp) or np.isnan(base_sl):
        continue

    for offset in range(1, lookahead+1):
        i = t + offset
        if i+1 >= N:
            break

        price_now = close_arr[i]
        sigma    = sigma_arr[i]
        if np.isnan(price_now) or np.isnan(sigma):
            continue

        sims    = simulate_close_numba(price_now, sigma, n_sim, dt=1.0)
        prob_tp = np.mean(sims >= base_tp)
        prob_sl = np.mean(sims <= base_sl)

        # esito reale sulla barra successiva
        h1, l1 = high_arr[i+1], low_arr[i+1]
        outcome = 1 if h1 >= base_tp else (-1 if l1 <= base_sl else 0)

        exploded.append({
            "prob_tp":    prob_tp,
            "prob_sl":    prob_sl,
            "outcome":    outcome
        })

# === 5. Costruzione array per il report ===
df_ex   = pl.DataFrame(exploded)
y_true  = (df_ex["outcome"] == 1).to_numpy().astype(int)   # 1=TP, 0=altro
prob_tp = df_ex["prob_tp"].to_numpy()

# === 6. Predizioni con soglia fissa ===
y_pred = (prob_tp >= threshold).astype(int)

# === 7. Classification report ===
print("\n🎯 CLASSIFICATION REPORT (TP vs non‐TP @ threshold={:.2f}):\n".format(threshold))
print(classification_report(y_true, y_pred, target_names=["non‐TP","TP"]))
