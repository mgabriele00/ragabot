import numpy as np
from numba import njit
import math
from service.analysis_service import calculate_max_drawdown_from_initial

@njit(fastmath=False)
def backtest(close, atr, signals, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee=np.float32(2.5)) -> np.ndarray:
    # Preallochiamo un array NumPy della dimensione corretta
    equity_curve = np.full(len(close), np.float32(0.0))
    
    realized_equity = np.float32(initial_equity)
    trade_count     = np.int32(0)
    gross_profit    = np.float32(0.0)
    gross_loss      = np.float32(0.0)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)
    #atr_threshold = np.float32(0.02)
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
         #   if atr_i < atr_threshold:
         #       continue
            position_open = True
            position_side = signal
            entry_price = price
            position_size = (exposure * realized_equity * leverage) / entry_price
            realized_equity -= np.float32(fixed_fee)
<<<<<<< Updated upstream
            #stop_loss = price - signal * sl_mult * atr_i
            stop_loss = price - signal * sl_mult
            take_profit = price + signal * tp_mult
=======
            stop_loss = price - signal * sl_mult * atr_i * 2
            take_profit = price + signal * tp_mult * atr_i * 4 #TODO: eliminare 2
>>>>>>> Stashed changes

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
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss   += pnl
            trade_count +=1
            realized_equity += pnl
            realized_equity -= np.float32(fixed_fee)
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
    max_dd = calculate_max_drawdown_from_initial(equity_curve, initial_equity)
    total_fees = trade_count * 2 * fixed_fee
    avg_pnl    = (gross_profit + gross_loss) / max(trade_count, 1)
    fee_ratio  = total_fees / max(gross_profit, np.float32(1.0))
    return equity_curve[-1], max_dd, trade_count, gross_profit, gross_loss, avg_pnl, total_fees, fee_ratio
