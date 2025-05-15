import numpy as np
from numba import njit


@njit(fastmath=True)
def simulate_close_numba(current_price, sigma, n_sim, dt=1 / 1440, mu=0.0):
    Z = np.random.normal(0, 1, n_sim)
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    return current_price * np.exp(log_ret)

@njit(fastmath=True)
def backtest(
        close, high, low, atr, sigma, signals,
        start_index, initial_equity,
        sl_mult, tp_mult, exposure, leverage,
        fixed_fee, lot_size,
        n_sim, threshold
) -> np.ndarray:
    equity_curve = np.full(len(close), np.float32(0.0))

    realized_equity = np.float32(initial_equity)
    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)
    entry_bar = np.int32(-1)

    for i in range(start_index, len(close) - 1):  # -1 perché simuliamo t+1
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]
        sigma_i = sigma[i]

        # 1. Apertura posizione
        if not position_open and signal != 0:
            position_open = True
            position_side = signal
            entry_price = price
            take_profit = entry_price + signal * tp_mult * atr_i
            stop_loss = entry_price - signal * sl_mult * atr_i
            position_size = (exposure * realized_equity * leverage) / entry_price
            realized_equity -= fixed_fee * position_size / lot_size
            entry_bar = i

        # 2. Simulazione per decidere uscita
        if position_open and i > entry_bar:
            if sigma_i > 0.0:
                sims = simulate_close_numba(price, sigma_i, n_sim)
                if position_side == 1:
                    prob_tp = np.mean(sims >= take_profit)
                    prob_sl = np.mean(sims <= stop_loss)
                else:
                    prob_tp = np.mean(sims <= take_profit)
                    prob_sl = np.mean(sims >= stop_loss)

                if prob_tp >= threshold:
                    exit_price = take_profit
                    position_open = False
                elif prob_sl >= threshold:
                    exit_price = stop_loss
                    position_open = False
                else:
                    exit_price = None
            else:
                exit_price = None
        else:
            exit_price = None

        # 3. Chiusura posizione
        if exit_price is not None:
            pnl = position_size * (exit_price - entry_price) * position_side
            realized_equity += pnl
            realized_equity -= fixed_fee * position_size / lot_size
            position_open = False
            position_side = 0
            entry_price = np.float32(0.0)
            stop_loss = np.float32(0.0)
            take_profit = np.float32(0.0)
            position_size = np.float32(0.0)
            entry_bar = -1

        # 4. Mark-to-market
        if position_open:
            unrealized = position_size * (price - entry_price) * position_side
        else:
            unrealized = 0.0

        current_equity = realized_equity + unrealized
        equity_curve[i] = current_equity

        # 5. Margin call
        if current_equity <= 0:
            equity_curve[i:] = 0.0
            break

    return equity_curve
