import numpy as np
from numba import njit
import math
from service.analysis_service import calculate_max_drawdown_from_initial


@njit(fastmath=True)
def backtest(close, high, low, atr, signals, start_index, initial_equity, sl_mult, tp_mult, exposure, leverage,
             fixed_fee, lot_size) -> np.ndarray:
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

    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]
        high_i = np.float32(high[i])
        low_i = np.float32(low[i])

        if not position_open and signal != 0:
            position_open = True
            position_side = signal
            entry_price = price
            position_size = (exposure * realized_equity * leverage) / entry_price
            realized_equity -= np.float32(fixed_fee) * position_size / lot_size
            stop_loss = price - signal * sl_mult * atr_i
            take_profit = price + signal * tp_mult * atr_i
            entry_bar = i

        exit_price = None
        if position_open:
            if position_side == 1 and signal == -1:
                exit_price = price
            elif position_side == -1 and signal == 1:
                exit_price = price
            else:
                if position_side == 1:
                    if price >= take_profit:
                        exit_price = price
                    elif price <= stop_loss:
                        exit_price = price
                else:
                    if price <= take_profit:
                        exit_price = price
                    elif price >= stop_loss:
                        exit_price = price

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

        if position_open:
            unrealized = position_size * (price - entry_price) * position_side
        else:
            unrealized = np.float32(0.0)

        current_equity = realized_equity + unrealized

        if position_open and i > entry_bar:
            if position_side == 1:
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

        equity_curve[i] = current_equity

    return equity_curve


@njit(fastmath=True)
def backtest_with_trades(close, high, low, atr, signals, rsi, bollinger_bands, bullish, bearish,
                         start_index, initial_equity, sl_mult, tp_mult, exposure, leverage, fixed_fee, lot_size):
    max_trades = len(close)
    entry_prices = np.full(max_trades, np.float32(0.0))
    exit_prices = np.full(max_trades, np.float32(0.0))
    entry_times = np.full(max_trades, np.int32(-1))
    exit_times = np.full(max_trades, np.int32(-1))
    pnls = np.full(max_trades, np.float32(0.0))
    trade_count = 0

    position_open = False
    position_side = 0
    entry_price = np.float32(0.0)
    stop_loss = np.float32(0.0)
    take_profit = np.float32(0.0)
    position_size = np.float32(0.0)

    for i in range(start_index, len(close)):
        price = np.float32(close[i])
        atr_i = np.float32(atr[i])
        signal = signals[i]

        if not position_open and signal != 0:
            position_open = True
            position_side = signal
            entry_price = price
            position_size = (exposure * initial_equity * leverage) / entry_price
            stop_loss = price - signal * sl_mult * atr_i
            take_profit = price + signal * tp_mult * atr_i

            entry_prices[trade_count] = entry_price
            entry_times[trade_count] = i

        exit_price = None
        exit_reason = 0

        if position_open:
            if (position_side == 1 and signal == -1) or (position_side == -1 and signal == 1):
                exit_price = price
                exit_reason = 1
            elif position_side == 1:
                if price >= take_profit:
                    exit_price = price
                    exit_reason = 2
                elif price <= stop_loss:
                    exit_price = price
                    exit_reason = 3
            else:
                if price <= take_profit:
                    exit_price = price
                    exit_reason = 2
                elif price >= stop_loss:
                    exit_price = price
                    exit_reason = 3

        if exit_price is not None:
            pnl = position_size * (exit_price - entry_price) * position_side - 2 * fixed_fee * position_size / lot_size

            exit_prices[trade_count] = exit_price
            exit_times[trade_count] = i
            pnls[trade_count] = pnl
            trade_count += 1

            position_open = False
            position_side = 0

    return (entry_prices[:trade_count], exit_prices[:trade_count],
            entry_times[:trade_count], exit_times[:trade_count],
            pnls[:trade_count])