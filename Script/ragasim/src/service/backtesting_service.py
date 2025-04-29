import numpy as np

from models.strategy_condition import StrategyCondition
from Script.ragasim.src.models.strategy_indicators import StrategyIndicators
from numba import njit

@njit(fastmath=True)
def get_tp_sl(atr, entry_price, sl_factor, tp_factor, position_type):
    if position_type == "long":
        sl = entry_price - (sl_factor * atr)
        tp = entry_price + (tp_factor * atr)
    elif position_type == "short":
        sl = entry_price + (sl_factor * atr)
        tp = entry_price - (tp_factor * atr)
    else:
        raise ValueError("Invalid position type. Use 'long' or 'short'.")
    return sl, tp

@njit(fastmath=True)
def backtest(signal: np.ndarray,
            close: np.ndarray,
            strategy_params: StrategyIndicators,
            strategy_condition: StrategyCondition,
            initial_cash: float,
            leverage: float,
            exposure: float
        ) -> float:
    """
        signal: array di -1/0/1
        close, atr: array dei prezzi e ATR
        sl_factor, tp_factor: fattori per SL/TP
        strategy_params, strategy_condition: oggetti numba
    initial_cash: capitale iniziale
    leverage: leva da applicare
    exposure: percentuale di equity da rischiare per trade (0-1)
        restituisce cash finale
        """
    cash = initial_cash
    position = None       # "long" / "short" / None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    units = 0.0

    atr_window = strategy_condition.atr_window
    sl_factor = strategy_condition.atr_factor
    tp_factor = strategy_condition.atr_factor
    # trova l’indice nella lista ATRParams corrispondente a questo atr_window
    atr_idx = 0
    for j, atrp in enumerate(strategy_params.atr):
        if atrp.window == atr_window:
            atr_idx = j
            break
    atr = strategy_params.atr[atr_idx].values
    for i in range(len(signal)):
        price = close[i]

        # verifica SL/TP
        if position == "long":
            if price <= stop_loss or price >= take_profit:
                pnl = units * (price - entry_price)
                cash += pnl
                position = None
        elif position == "short":
            if price >= stop_loss or price <= take_profit:
                pnl = units * (entry_price - price)
                cash += pnl
                position = None

        sig = signal[i]
        # entry / exit logic
        if sig == 1:  # buy
            if position is None:
                entry_price = price
                units = (cash * leverage * exposure) / price
                stop_loss, take_profit = get_tp_sl(atr[i], price, sl_factor, tp_factor, "long")
                position = "long"
            elif position == "short":
                pnl = units * (entry_price - price)
                cash += pnl
                position = None

        elif sig == -1:  # sell
            if position is None:
                entry_price = price
                units = (cash * leverage * exposure) / price
                stop_loss, take_profit = get_tp_sl(atr[i], price, sl_factor, tp_factor, "short")
                position = "short"
            elif position == "long":
                pnl = units * (price - entry_price)
                cash += pnl
                position = None

    return cash