import time
from datetime import datetime
import talib
import numpy as np
import MetaTrader5 as mt5
from datetime import timedelta

print("→ [Module] trading_bot.py loaded")

# ─── costanti e parametri ─────────────────────────────────────────────────────
LEVAREGE   = 100        # leva del broker
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

def compute_volume(balance, exposure, leverage):
    # 1 lot = 100 000 base currency
    lots = (balance * exposure * leverage) / 100000
    return round(lots, 2)

def generate_signals(rates, params: dict):
    """
    rates: lista di struct MT5 con attributi .open, .high, .low, .close
    """
    # estrai vettori da structured array
    close = np.array(rates['close'], dtype=float)
    open_ = np.array(rates['open'],  dtype=float)
    high  = np.array(rates['high'],  dtype=float)
    low   = np.array(rates['low'],   dtype=float)

    # RSI
    rsi = talib.RSI(close, timeperiod=14)

    # pattern recognition
    bullish = np.zeros_like(close, dtype=bool)
    bearish = np.zeros_like(close, dtype=bool)
    for pat in talib.get_function_groups()["Pattern Recognition"]:
        vals = getattr(talib, pat)(open_, high, low, close)
        bullish |= (vals > 0)
        bearish |= (vals < 0)

    # Bollinger Bands
    upper, _, lower = talib.BBANDS(
        close,
        timeperiod=14,
        nbdevup=params["bb_std"],
        nbdevdn=params["bb_std"]
    )

    # vettorializza segnali
    entries = (rsi < params["rsi_entry"]) & (close < lower) & bullish
    exits   = (rsi > params["rsi_exit"])  & (close > upper) & bearish

    print(f"   Signals: entries={np.count_nonzero(entries)}, exits={np.count_nonzero(exits)}")
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
    
    last_min = None

    try:
        while True:
            now = datetime.now()
            if now.minute != last_min:
                last_min = now.minute
                print(f"\n[{now:%Y-%m-%d %H:%M:%S}] === New minute tick ===")
                # 1) scarica dati da inizio anno
                start = datetime.now() - timedelta(hours=1)
                rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start, now)
                if len(rates) == 0:
                    print("   [Warning] Nessun dato ricevuto, salto ciclo")
                    time.sleep(1)
                    continue
                print(f"   [Data] Ricevuti {len(rates)} barre")

                # 2) segnali direttamente da rates
                entries, exits = generate_signals(rates, PARAMS)

                # 3) calcola volume
                info    = mt5.account_info()
                balance = info.balance
                vol     = compute_volume(balance, PARAMS["exposure"], LEVAREGE)
                print(f"   Balance={balance:.2f}, exposure={PARAMS['exposure']} → volume={vol} lot")

                # 4) prendi prezzi correnti
                tick = mt5.symbol_info_tick(SYMBOL)
                ask, bid = tick.ask, tick.bid
                print(f"   Prices: ask={ask:.5f}, bid={bid:.5f}")

                # 5) gestione posizioni long only
                positions = mt5.positions_get(symbol=SYMBOL)
                if not positions:
                    # se non ho posizioni aperte apro LONG solo su entry
                    if entries[-1]:
                        print("   [Decision] Entry signal → BUY")
                        slp = ask * (1 - PARAMS["sl"])
                        tpp = ask * (1 + PARAMS["tp"])
                        place_order(mt5.ORDER_TYPE_BUY, ask, slp, tpp, vol)
                    else:
                        print("   [Info] Nessuna posizione aperta e nessun entry signal")
                else:
                    # se ho già 1+ posizioni long, chiudo se scatta exit
                    if exits[-1]:
                        print("   [Decision] Exit signal → CLOSING LONG")
                        for pos in positions:
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                close_req = {
                                    "action":    mt5.TRADE_ACTION_DEAL,
                                    "symbol":    SYMBOL,
                                    "volume":    pos.volume,
                                    "type":      mt5.ORDER_TYPE_SELL,
                                    "position":  pos.ticket,
                                    "price":     bid,
                                    "deviation": DEVIATION,
                                    "magic":     MAGIC,
                                    "comment":   "auto_strategy close"
                                }
                                res = mt5.order_send(close_req)
                                print(f"   → close result: {res}")
                    else:
                        print("   [Info] Posizione long aperta, nessun exit signal")
            time.sleep(1)
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()