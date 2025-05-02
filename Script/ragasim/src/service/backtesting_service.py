import numpy as np
from numba import njit
import math

@njit(fastmath=False)
def backtest(close, atr, signals, initial_equity, sl_mult, tp_mult, exposure, leverage) -> np.ndarray:
    # Preallochiamo un array NumPy della dimensione corretta
    equity_curve = np.full(len(close), np.float32(0.0))
    
    realized_equity = np.float32(initial_equity)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)

    # 1. Pre-loop: barre con ATR NaN
    start_index = 0
    while start_index < len(atr) and math.isnan(atr[start_index]):
        equity_curve[start_index] = realized_equity
        start_index += 1

    # 2. Loop principale
    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]

        # 2.1 Apertura posizione
        if not position_open and signal != 0:
            position_open = True
            position_side = signal
            entry_price = price
            position_size = (exposure * realized_equity * leverage) / entry_price
            stop_loss = price - signal * sl_mult * atr_i
            take_profit = price + signal * tp_mult * atr_i * 2 #TODO: eliminare 2

        # 2.2 Controllo uscita
        exit_price = None
        if position_open:
            # uscita per inversione di segnale
            if position_side == 1 and signal == -1:
                exit_price = price
            elif position_side == -1 and signal == 1:
                exit_price = price
            else:
                # uscita per TP/SL
                if position_side == 1:
                    if price >= take_profit:
                        exit_price = take_profit
                    elif price <= stop_loss:
                        exit_price = stop_loss
                else:
                    if price <= take_profit:
                        exit_price = take_profit
                    elif price >= stop_loss:
                        exit_price = stop_loss

        # 2.3 Realizza PnL se serve
        if exit_price is not None:
            pnl = position_size * (exit_price - entry_price) * position_side
            realized_equity += pnl
            position_open = False
            position_side = 0
            entry_price = np.float32(0.0)
            stop_loss = np.float32(0.0)
            take_profit = np.float32(0.0)
            position_size = np.float32(0.0)

        # 2.4 Mark-to-market intrabar
        if position_open:
            unrealized = position_size * (price - entry_price) * position_side
        else:
            unrealized = np.float32(0.0)

        current_equity = realized_equity + unrealized

        # 2.5 Bancarotta: fill-zero e break
        if current_equity <= 0:
            # Riempi tutto il resto dell'array con zeri
            equity_curve[i:] = np.float32(0.0)
            break

        # 2.6 Registra equity di fine barra
        equity_curve[i] = current_equity
    return equity_curve[-1]#equity_curve.astype(np.float32)
