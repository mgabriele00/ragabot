import numpy as np
from numba import njit
import math
from service.analysis_service import calculate_max_drawdown_from_initial

@njit(fastmath=True)
def backtest(close, high, low, atr, signals, start_index, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee, lot_size) -> np.ndarray:
    # Preallochiamo un array NumPy della dimensione corretta
    equity_curve = np.full(len(close), np.float32(0.0))
    
    realized_equity = np.float32(initial_equity)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)
    entry_bar = np.int32(-1)
    
    # 2. Loop principale
    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]
        high_i = np.float32(high[i])
        low_i = np.float32(low[i])

        
        # 2.1 Apertura posizione
        if not position_open and signal != 0:
            position_open = True
            position_side = signal
            entry_price = price
            position_size = (exposure * realized_equity * leverage) / entry_price
            realized_equity -= np.float32(fixed_fee) * position_size / lot_size
            stop_loss = price - signal * sl_mult * atr_i
            take_profit = price + signal * tp_mult * atr_i
            entry_bar = i

        # 2.2 Controllo uscita
        exit_price = None
        if position_open and i > entry_bar :
            # uscita per inversione di segnale
            if position_side == 1 and signal == -1:
                exit_price = price
            elif position_side == -1 and signal == 1:
                exit_price = price
            else:
                # uscita per TP/SL
                if position_side == 1:
                    if high_i >= take_profit:
                        exit_price = take_profit
                    elif low_i <= stop_loss:
                        exit_price = stop_loss
                else:
                    if low_i <= take_profit:
                        exit_price = take_profit
                    elif high_i >= stop_loss:
                        exit_price = stop_loss

        # 2.3 Realizza PnL se serve
        if exit_price is not None:
            pnl = position_size * (exit_price - entry_price) * position_side
            realized_equity += pnl
            realized_equity -= np.float32(fixed_fee) * position_size / lot_size
            position_open = False
            position_side = 0
            entry_price = np.float32(0.0)
            stop_loss = np.float32(0.0)
            take_profit = np.float32(0.0)
            position_size = np.float32(0.0)
            entry_bar = np.int32(-1)

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
            
    return equity_curve