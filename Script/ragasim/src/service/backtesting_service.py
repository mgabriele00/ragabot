import numpy as np

from models.strategy_condition import StrategyCondition
from models.strategy_indicators import StrategyIndicators
from numba import njit

@njit(fastmath=True)
def get_tp_sl(atr, entry_price, sl_factor, tp_factor, position_type):
    # Verifica ATR non valido prima di qualsiasi calcolo
    if atr <= 0:
        return np.nan, np.nan

    if position_type == "long":
        sl = entry_price - sl_factor * atr
        tp = entry_price + tp_factor * atr
    elif position_type == "short":
        sl = entry_price + sl_factor * atr
        tp = entry_price - tp_factor * atr
    else:
        return np.nan, np.nan

    return sl, tp

@njit(fastmath=True)
def calculate_max_drawdown_numba(equity_curve: np.ndarray) -> float:
    """Calcola il Maximum Drawdown in percentuale (0-100)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        if equity <= 0:
            max_dd = max(max_dd, 1.0)
    return max_dd * 100.0

@njit(fastmath=True)
def backtest(
    signal: np.ndarray,
    close: np.ndarray,
    strategy_indicators: StrategyIndicators,
    strategy_condition: StrategyCondition,
    initial_cash: float,
    leverage: float,
    exposure: float
) -> tuple[float, float, float, float]:
    """
    Esegue il backtest di una strategia Forex con ATR-based SL/TP e leva.

    Restituisce:
      final_equity: equity finale (>= 0)
      total_pnl: PnL netto = final_equity - initial_cash
      max_drawdown: max drawdown percentuale (0-100)
      win_rate: percentuale di trade vincenti (0-100)
    """
    # --- Controllo parametri iniziali ---
    if np.isnan(initial_cash) or np.isnan(leverage) or np.isnan(exposure) or \
       np.isnan(strategy_condition.atr_window) or np.isnan(strategy_condition.atr_factor):
        return initial_cash, 0.0, 0.0, 0.0

    cash = initial_cash
    margin = 0.0
    position = 0
    entry_price = 0.0
    stop_loss = np.nan
    take_profit = np.nan
    units = 0.0
    total_trades = 0
    winning_trades = 0

    # Curva equity per drawdown
    equity_curve = np.full(len(signal), np.nan)

    atr_window = strategy_condition.atr_window
    sl_factor = strategy_condition.atr_factor
    tp_factor = strategy_condition.atr_factor * 2

    # Estrae ATR corrispondente
    atr_idx = -1
    for j in range(len(strategy_indicators.atr)):
        if strategy_indicators.atr[j].window == atr_window:
            atr_idx = j
            break
    if atr_idx == -1:
        return initial_cash, 0.0, 0.0, 0.0
    atr = strategy_indicators.atr[atr_idx].values

    # Primo indice valido ATR
    first_valid_idx = -1
    for k in range(len(atr)):
        if not np.isnan(atr[k]):
            first_valid_idx = k
            break
    if first_valid_idx == -1:
        return initial_cash, 0.0, 0.0, 0.0

    # Inizializza equity_curve al primo step
    if first_valid_idx > 0:
        equity_curve[first_valid_idx - 1] = initial_cash
    else:
        equity_curve[0] = initial_cash

    # Loop di backtest
    for i in range(first_valid_idx, len(signal)):
        price = close[i]
        current_atr = atr[i]

        # Equity mark-to-market
        unrealized_pnl = 0.0
        if position == 1:
            unrealized_pnl = units * (price - entry_price)
        elif position == -1:
            unrealized_pnl = units * (entry_price - price)
        current_equity = cash + unrealized_pnl
        equity_curve[i] = max(current_equity, 0.0)

        # Bancarotta
        if current_equity <= 0:
            equity_curve[i] = 0.0
            break

        # Verifica SL/TP
        close_reason = 0
        trade_pnl = 0.0
        if position == 1:
            if not np.isnan(stop_loss) and price <= stop_loss:
                trade_pnl = units * (stop_loss - entry_price)
                close_reason = 1
            elif not np.isnan(take_profit) and price >= take_profit:
                trade_pnl = units * (take_profit - entry_price)
                close_reason = 2
        elif position == -1:
            if not np.isnan(stop_loss) and price >= stop_loss:
                trade_pnl = units * (entry_price - stop_loss)
                close_reason = 1
            elif not np.isnan(take_profit) and price <= take_profit:
                trade_pnl = units * (entry_price - take_profit)
                close_reason = 2

        if close_reason > 0:
            # Ripristina margin e aggiunge PnL
            cash += margin + trade_pnl
            margin = 0.0
            total_trades += 1
            if trade_pnl > 0:
                winning_trades += 1
            position = 0
            units = 0.0
            entry_price = 0.0
            stop_loss = np.nan
            take_profit = np.nan
            current_equity = cash
            equity_curve[i] = max(current_equity, 0.0)

        # Uscita per segnale opposto
        sig = signal[i]
        if position != 0 and close_reason == 0:
            exit_pnl = 0.0
            if (sig == -1 and position == 1) or (sig == 1 and position == -1):
                if position == 1:
                    exit_pnl = units * (price - entry_price)
                else:
                    exit_pnl = units * (entry_price - price)
                cash += margin + exit_pnl
                margin = 0.0
                total_trades += 1
                if exit_pnl > 0:
                    winning_trades += 1
                position = 0
                units = 0.0
                entry_price = 0.0
                stop_loss = np.nan
                take_profit = np.nan
                current_equity = cash
                equity_curve[i] = max(current_equity, 0.0)

        # Ingresso
        if sig != 0 and position == 0 and close_reason == 0 and current_atr > 0 and current_equity > 0:
            entry_price = price
            units = (current_equity * leverage * exposure) / price if price > 0 else 0.0
            if units > 0:
                sl, tp = get_tp_sl(current_atr, price, sl_factor, tp_factor,
                                    "long" if sig == 1 else "short")
                if not (np.isnan(sl) or np.isnan(tp)):
                    stop_loss = sl
                    take_profit = tp
                    # Blocca margin
                    margin = (units * entry_price) / leverage
                    cash -= margin
                    position = sig
                else:
                    units = 0.0
                    entry_price = 0.0
            else:
                units = 0.0
                entry_price = 0.0

    # Calcoli finali
    final_equity = cash
    last_price = close[-1]
    if position == 1:
        final_equity = cash + units * (last_price - entry_price)
    elif position == -1:
        final_equity = cash + units * (entry_price - last_price)
    final_equity = max(final_equity, 0.0)

    max_drawdown = calculate_max_drawdown_numba(equity_curve)
    total_pnl = final_equity - initial_cash
    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

    return final_equity, total_pnl, max_drawdown, win_rate
