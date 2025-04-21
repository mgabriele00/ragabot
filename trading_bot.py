import sys
import time
from datetime import datetime, date
import numpy as np
import pandas as pd
import talib
import MetaTrader5 as mt5
from datetime import timedelta

print("→ [Module] trading_bot.py loaded")

# ─── costanti e parametri ─────────────────────────────────────────────────────
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
DEVIATION = 10        # slippage massimo
MAGIC     = 123456    # magic number
PARAMS = {
    "sl":       0.006,   # 0.6%
    "tp":       0.02,    # 2%
    "rsi_entry":35,
    "rsi_exit": 55,
    "bb_std":   1.75,
    "exposure": 0.6      # 60% del balance
}

def init_mt5():
    ok = mt5.initialize()
    if not ok:
        print("MT5 initialize failed:", mt5.last_error())
        raise SystemExit("❌ Impossibile inizializzare MT5: controlla che il terminal sia aperto e la Python API abilitata.")
    print("✅ MT5 initialized, version:", mt5.version())

def compute_volume(balance, exposure):
    # 1 lot = 100 000 base currency
    lots = (balance * exposure) / 100000
    return round(lots, 2)

def calculate_indicators(df: pd.DataFrame, params: dict):
    """
    df: pandas DataFrame con colonne 'Open','High','Low','Close'
    """
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    # 1) RSI
    rsi = talib.RSI(close, timeperiod=14)
    # 2) Pattern Recognition
    bullish = np.zeros_like(close, dtype=bool)
    bearish = np.zeros_like(close, dtype=bool)
    for pat in talib.get_function_groups()["Pattern Recognition"]:
        vals = getattr(talib, pat)(open_, high, low, close)
        bullish |= vals > 0
        bearish |= vals < 0
    # 3) Bollinger Bands
    upper, _, lower = talib.BBANDS(
        close,
        timeperiod=14,
        nbdevup=params["bb_std"],
        nbdevdn=params["bb_std"]
    )
    return close, rsi, bullish, bearish, upper, lower

def generate_signals(df: pd.DataFrame, params: dict):
    close, rsi, bullish, bearish, upper, lower = calculate_indicators(df, params)
    entries = (rsi < params["rsi_entry"]) & (close < lower) & bullish
    exits   = (rsi > params["rsi_exit"])  & (close > upper) & bearish
    print(f"   Signals: entries={entries.sum()}, exits={exits.sum()}")
    print(f"   Latest tick: entry={bool(entries[-1])}, exit={bool(exits[-1])}")
    return entries, exits

def place_order(order_type, price, sl_price, tp_price, volume):
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    DEVIATION,
        "magic":        MAGIC,
        "comment":      "auto_strategy",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    res = mt5.order_send(req)
    side = "BUY" if order_type==mt5.ORDER_TYPE_BUY else "SELL"
    print(f"{datetime.now():%H:%M:%S} → {side} {volume} @ {price:.5f} SL={sl_price:.5f} TP={tp_price:.5f} → {res}")

def main():
    print("→ [Main] invoking main()")
    try:
        init_mt5()
    except SystemExit as e:
        print(e)
        return

    start = datetime.now() - timedelta(hours=1)
    last_min = None

    try:
        while True:
            now = datetime.now()
            if now.minute != last_min:
                last_min = now.minute
                print(f"\n[{now:%Y-%m-%d %H:%M:%S}] === New minute tick ===")
                # 1) scarica dati da inizio anno
                rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start, now)
                if len(rates) == 0:
                    print("   [Warning] Nessun dato ricevuto, salto ciclo")
                    time.sleep(1)
                    continue
                print(f"   [Data] Ricevuti {len(rates)} barre")

                # 2) dataframe + segnali (usando pandas)
                df = pd.DataFrame(rates)
                df['time']=pd.to_datetime(df['time'], unit='s')

                entries, exits = generate_signals(df, PARAMS)

                # 3) calcola volume
                info    = mt5.account_info()
                balance = info.balance
                vol     = compute_volume(balance, PARAMS["exposure"])
                print(f"   Balance={balance:.2f}, exposure={PARAMS['exposure']} → volume={vol} lot")

                # 4) prendi prezzi correnti
                tick = mt5.symbol_info_tick(SYMBOL)
                ask, bid = tick.ask, tick.bid
                print(f"   Prices: ask={ask:.5f}, bid={bid:.5f}")

                # 5) invia ordine
                sl_pct = PARAMS["sl"]
                tp_pct = PARAMS["tp"]
                if entries[-1]:
                    print("   [Decision] Entry signal → BUY")
                    slp = ask * (1 - sl_pct)
                    tpp = ask * (1 + tp_pct)
                    place_order(mt5.ORDER_TYPE_BUY, ask, slp, tpp, vol)
                elif exits[-1]:
                    print("   [Decision] Exit signal → SELL")
                    slp = bid * (1 + sl_pct)
                    tpp = bid * (1 - tp_pct)
                    place_order(mt5.ORDER_TYPE_SELL, bid, slp, tpp, vol)
            time.sleep(1)
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()