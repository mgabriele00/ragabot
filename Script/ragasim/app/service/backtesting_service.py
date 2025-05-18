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
    position_size = np.float32(0.0)
    entry_bar = np.int32(-1)
    
    sl_count = 0
    perdite = np.zeros(len(close), dtype=np.float32)
    guadagno = np.zeros(len(close), dtype=np.float32)
    
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
                # uscita per SL
                if position_side == 1:
                    if price <= stop_loss:
                        sl_count+=1
                        exit_price = stop_loss 
                else:
                    if price >= stop_loss:
                        sl_count+=1
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
            position_size = np.float32(0.0)
            entry_bar = np.int32(-1)
            if pnl < 0:
                perdite[i] = pnl
            else:
                guadagno[i] = pnl

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
    print("SL count: ", sl_count)
    #array di percentuale di perdite rispetto all'equity 
    perdite_perc = (perdite / realized_equity) * 100
    guadagno_perc = (guadagno / realized_equity) * 100
    print("Perdite in percentuale: ", np.mean(perdite_perc[perdite_perc != 0]))
    print("Guadagni in percentuale: ", np.mean(guadagno_perc[guadagno_perc != 0]))
    
    print("Guadagno Medio: ", np.mean(guadagno[guadagno != 0]))
    print("Perdita Media: ", np.mean(perdite[perdite != 0]))
    n_guadagni = np.count_nonzero(guadagno)
    n_perdite = np.count_nonzero(perdite)
    print("Numero di perdite: ", n_perdite)
    print("Numero di guadagni: ", n_guadagni)
            
    return equity_curve