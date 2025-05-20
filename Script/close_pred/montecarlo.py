import polars as pl
import numpy as np
from ta.volatility import AverageTrueRange
from numba import njit
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt

# === Parametri ===
tp_mult      = 0.1
sl_mult      = 20
n_sim        = 10000
lookahead    = 10
window_sigma = 100

# === Soglie manuali (per report finale) ===
thr_tp = 0.7
thr_sl = 0.7

# === 1. Caricamento e parsing datetime ===
df_raw = pl.read_csv("Script/close_pred/data/EURUSD_M1_2013_2024.csv")
df_raw = df_raw.with_columns(
    (pl.col("Date") + pl.lit(" ") + pl.col("Time"))
      .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M", strict=False)
      .alias("datetime")
).drop(["Date","Time"]).sort("datetime")
df_raw = df_raw.rename({
    "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
})

# === 2. ATR e indicatori via pandas ===
df_pd = df_raw.to_pandas()
atr = AverageTrueRange(high=df_pd["high"], low=df_pd["low"], close=df_pd["close"], window=14).average_true_range()
df_pd["atr"]     = atr
df_pd["log_ret"] = np.log(df_pd["close"]/df_pd["close"].shift(1))
df_pd["sigma"]   = df_pd["log_ret"].rolling(window_sigma).std()
df_pd["tp"]      = df_pd["close"] + atr * tp_mult
df_pd["sl"]      = df_pd["close"] - atr * sl_mult
df = pl.from_pandas(df_pd).drop_nulls()

# === 3. Simulatore Monte Carlo ===
@njit(fastmath=True)
def simulate_close_numba(current_price, sigma, n_sim, dt=1.0, mu=0.0):
    Z = np.random.randn(n_sim)
    log_ret = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    return current_price * np.exp(log_ret)

# === 4. Esplodi e simula sulle prime 100 barre ===
exploded = []
close_arr = df["close"].to_numpy()
high_arr  = df["high"].to_numpy()
low_arr   = df["low"].to_numpy()
tp_arr    = df["tp"].to_numpy()
sl_arr    = df["sl"].to_numpy()
sigma_arr = df["sigma"].to_numpy()

N = min(df.height, df.height)
for t in range(N - lookahead - 1):
    base_tp = tp_arr[t]
    base_sl = sl_arr[t]
    if np.isnan(base_tp) or np.isnan(base_sl):
        continue
    for offset in range(1, lookahead+1):
        i = t + offset
        if i+1 >= N: break
        price, sigma = close_arr[i], sigma_arr[i]
        if np.isnan(price) or np.isnan(sigma): continue
        sims    = simulate_close_numba(price, sigma, n_sim)
        prob_tp = np.mean(sims >= base_tp)
        prob_sl = np.mean(sims <= base_sl)
        h1, l1 = high_arr[i+1], low_arr[i+1]
        outcome = 1 if h1>=base_tp else (-1 if l1<=base_sl else 0)
        exploded.append({"prob_tp":prob_tp, "prob_sl":prob_sl, "outcome":outcome})

# === 5. Costruisci vettori ===
import polars as pl
df_ex = pl.DataFrame(exploded)
y_true = df_ex["outcome"].to_numpy().astype(int)
prob_tp = df_ex["prob_tp"].to_numpy()
prob_sl = df_ex["prob_sl"].to_numpy()

# binarizza per TP e SL
y_true_tp = (y_true == 1).astype(int)
y_true_sl = (y_true == -1).astype(int)

# === 6. Calcola precision, recall, thresholds e F1 per TP e SL ===
prec_tp, rec_tp, thr_tp_pr = precision_recall_curve(y_true_tp, prob_tp)
prec_sl, rec_sl, thr_sl_pr = precision_recall_curve(y_true_sl, prob_sl)

# F1 score arrays (len = len(prec)-1)
f1_tp = 2 * (prec_tp[:-1] * rec_tp[:-1]) / (prec_tp[:-1] + rec_tp[:-1] + 1e-9)
f1_sl = 2 * (prec_sl[:-1] * rec_sl[:-1]) / (prec_sl[:-1] + rec_sl[:-1] + 1e-9)

# === 7. Plot F1 vs threshold ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.plot(thr_tp_pr, f1_tp, marker=".")
ax1.set_title("F1 score vs Threshold (TP_hit)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("F1 score")
ax1.axvline(thr_tp, color="gray", ls="--", label=f"manual thr={thr_tp}")
ax1.legend()

ax2.plot(thr_sl_pr, f1_sl, marker=".", color="orange")
ax2.set_title("F1 score vs Threshold (SL_hit)")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("F1 score")
ax2.axvline(thr_sl, color="gray", ls="--", label=f"manual thr={thr_sl}")
ax2.legend()

plt.tight_layout()
plt.show()

# === 8. Report multiclass con soglie manuali ===
y_pred = np.zeros_like(y_true)
y_pred[prob_tp >= thr_tp] = 1
mask_sl = (y_pred==0) & (prob_sl>=thr_sl)
y_pred[mask_sl] = -1

print(f"\n🎯 REPORT multiclass (prime 100 barre):\n")
print(classification_report(
    y_true, y_pred,
    labels=[-1,0,1],
    target_names=["SL_hit","No_hit","TP_hit"]
))
