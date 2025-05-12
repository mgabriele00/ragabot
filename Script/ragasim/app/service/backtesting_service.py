import numpy as np
from numba import njit
import math
from service.analysis_service import calculate_max_drawdown_from_initial

@njit(fastmath=True)
def backtest(close, high, low, atr, signals, start_index, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee, lot_size, waiting_number) -> np.ndarray:
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
    waiting_for_exit = False
    exit_price = None
    waiting_count = 0
    
    # 2. Loop principale
    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]
        high_i = np.float32(high[i])
        low_i = np.float32(low[i])
        
        skip_exit_checks = False
        
        if waiting_for_exit and position_open and exit_price is not None:
                # Se la posizione è aperta, controlla se il prezzo ha toccato il take profit o lo stop loss
                if low_i <= exit_price <= high_i:
                    waiting_for_exit = False
                    waiting_count = 0
                else:
                    # Se il prezzo non ha toccato il take profit o lo stop loss, incrementa il contatore di attesa
                    waiting_count += 1
                    if waiting_count >= waiting_number:
                        # Se il contatore di attesa supera N barre, esci dalla posizione
                        exit_price = price
                        waiting_for_exit = False
                        waiting_count = 0
                        skip_exit_checks = True
        
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
        if position_open and not skip_exit_checks:
            # uscita per inversione di segnale
            if position_side == 1 and signal == -1:
                exit_price = price
                waiting_for_exit = False
            elif position_side == -1 and signal == 1:
                exit_price = price
                waiting_for_exit = False
            elif not waiting_for_exit:
                # uscita per TP/SL
                if position_side == 1:
                    if price >= take_profit:
                        exit_price = take_profit
                        waiting_for_exit = True
                    elif price <= stop_loss:
                        exit_price = stop_loss
                        waiting_for_exit = True
                else:
                    if price <= take_profit:
                        exit_price = take_profit
                        waiting_for_exit = True
                    elif price >= stop_loss:
                        exit_price = price
                        waiting_for_exit = True

                        
        # 2.3 Realizza PnL se serve
        if exit_price is not None and not waiting_for_exit:
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
            exit_price = None
            waiting_for_exit = False

        # 2.4 Mark-to-market intrabar
        if position_open:
            unrealized = position_size * (price - entry_price) * position_side
        else:
            unrealized = np.float32(0.0)

        current_equity = realized_equity + unrealized

        if position_open and i>entry_bar:
            if position_side==1:
                terminal_exit = low_i
            else:
                terminal_exit = high_i
            
            terminal_pnl = position_size * (terminal_exit - entry_price) * position_side
            terminal_fee = np.float32(fixed_fee) * position_size / lot_size
            terminal_equity = realized_equity + terminal_pnl - terminal_fee
        else:
            terminal_equity = realized_equity
            
        if terminal_equity <= 0:
            equity_curve[i:] = np.float32(0.0)
            break

        # 2.6 Registra equity di fine barra
        equity_curve[i] = current_equity
            
    return equity_curve