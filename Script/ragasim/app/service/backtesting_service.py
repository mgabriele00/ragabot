import numpy as np
from numba import njit
from service.analysis_service import calculate_max_drawdown_from_initial

@njit(fastmath=True)
def simulate_close_numba(prev_price, sigma) -> np.ndarray:
    n_sim = 10000
    dt = 1 / 1440
    mu = 0.0
    Z = np.random.normal(0, 1, n_sim)
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return prev_price * np.exp(log_ret)

@njit(fastmath=True)
def hit_tp(target_price, prev_price, threshold, sigma, is_long) -> bool:
    sim_prices = simulate_close_numba(prev_price, sigma)
    probability = np.mean(sim_prices >= target_price) if is_long else np.mean(sim_prices <= target_price)
    return probability >= threshold

@njit(fastmath=True)
def hit_sl(target_price, prev_price, threshold, sigma, is_long) -> bool:
    sim_prices = simulate_close_numba(prev_price, sigma)
    probability = np.mean(sim_prices <= target_price) if is_long else np.mean(sim_prices >= target_price)
    return probability >= threshold

@njit(fastmath=True)
def calculate_sigma(close: np.ndarray) -> np.ndarray:
    window_sigma = 100
    # 1. Log return
    log_ret = np.full_like(close, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # 2. Rolling std su log return (window_sigma = 100)
    sigma = np.full_like(log_ret, np.nan)
    for i in range(window_sigma, len(log_ret)):
        sigma[i] = np.std(log_ret[i - window_sigma:i])
    return sigma

@njit(fastmath=True)
def backtest(close, atr, signals, start_index, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee, lot_size) -> np.ndarray:
    # Preallochiamo un array NumPy della dimensione corretta
    equity_curve = np.full(len(close), np.float32(0.0))
    
    realized_equity = np.float32(initial_equity)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)

    sigma = calculate_sigma(close)
    threshold_ = 0.7
    # 2. Loop principale
    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        prev_price = np.float32(close[i - 1])
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
            take_profit = price + signal * tp_mult * atr_i

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
                    if hit_tp(take_profit, prev_price, threshold_, sigma[i], True):
                        exit_price = take_profit
                    elif hit_sl(stop_loss, prev_price, threshold_, sigma[i], True):
                        exit_price = stop_loss
                else:
                    if hit_tp(take_profit, prev_price, threshold_, sigma[i], False):
                        exit_price = take_profit
                    elif hit_sl(stop_loss, prev_price, threshold_, sigma[i], False):
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