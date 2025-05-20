import numpy as np
from numba import njit, prange
from service.analysis_service import calculate_max_drawdown_from_initial

def hit_stop_loss(close, stop_loss, isLong) -> bool:
    return close <= stop_loss if isLong else close >= stop_loss

def backtest(close, low, high,  atr, signals, start_index, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee, lot_size, window, threshold, alpha) -> np.ndarray:
    equity_curve = np.full(len(close), np.float32(0.0))
    
    realized_equity = np.float32(initial_equity)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    position_size = np.float32(0.0)
        
    sl_ingarrati = 0
    sl_intercettati = 0
    sl_totali = 0
    
    count_stop_loss_arr = []
    count_stop_loss = 0
    
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
            realized_equity -= np.float32(fixed_fee) * position_size / lot_size
            stop_loss = price - signal * sl_mult * atr_i

        # 2.2 Controllo uscita
        exit_price = None
        if position_open:
            # uscita per inversione di segnale
            if hit_stop_loss(close[i], stop_loss, position_side == 1):
                sl_intercettati += 1
                if position_side == 1 and low[i] <=  stop_loss and stop_loss <= close[i] <= high[i]:
                    exit_price = stop_loss
                    sl_ingarrati += 1
                elif position_side == -1 and high[i] >= stop_loss and stop_loss >= low[i]:
                    exit_price = stop_loss
                    sl_ingarrati += 1    
                #else: exit_price = price
            elif position_side == 1 and signal == -1:
                exit_price = price
                count_stop_loss = 0
            elif position_side == -1 and signal == 1:
                exit_price = price
                count_stop_loss = 0
            else: count_stop_loss += 1
            
            if position_side == 1 and close[i] <= stop_loss:
                sl_totali += 1
            elif position_side == -1 and close[i] >= stop_loss:
                sl_totali += 1

        # 2.3 Realizza PnL se serve
        if exit_price is not None:
            pnl = position_size * (exit_price - entry_price) * position_side
            realized_equity += pnl
            realized_equity -= np.float32(fixed_fee) * position_size / lot_size
            position_open = False
            position_side = 0
            entry_price = np.float32(0.0)
            stop_loss = np.float32(0.0)
            position_size = np.float32(0.0)
            if count_stop_loss > 0:
                count_stop_loss_arr.append(count_stop_loss)
                count_stop_loss = 0

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
        
    sl_ingarrati_percent = np.float32(0.0)
    # Se vuoi comunque una percentuale, puoi moltiplicare per 100:
    if sl_totali > 0:
        sl_ingarrati_percent = np.float32(sl_ingarrati) / np.float32(sl_intercettati) * 100.0
        print("SL ingarrati perc:", sl_ingarrati_percent)
        print("SL intercettati:", sl_intercettati)
        print("SL totali:", sl_totali)
    else: print("SL ingarrati: 0.0")     
    
    count_stop_loss_arr = np.array(count_stop_loss_arr)
    count_stop_loss_mean = np.mean(count_stop_loss_arr) if len(count_stop_loss_arr) > 0 else 0
    print("Count stop loss mean:", count_stop_loss_mean)  
              
    return equity_curve