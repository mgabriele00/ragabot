import numpy as np
from numba import njit
from models.strategy_indicators import StrategyIndicators # Assicurati che gli import siano corretti
from models.strategy_condition import StrategyCondition

@njit
def get_signal(strategy_indicators, strategy_condition) -> np.ndarray:
    rsi      = strategy_indicators.rsi
    bullish  = strategy_indicators.bullish
    bearish  = strategy_indicators.bearish
    bb_std   = strategy_condition.bb_std
    bb_width_threshold = strategy_condition.bb_width_threshold

    idx = 0
    tol = 0.005  # metà del passo di 2 decimali
    for j in range(len(strategy_indicators.bollinger)):
        bb_val = strategy_indicators.bollinger[j].bb_std[0]
        if abs(bb_val - strategy_condition.bb_std) < tol:
            idx = j
            break
        
    upper = strategy_indicators.bollinger[idx].upper
    lower = strategy_indicators.bollinger[idx].lower
    bb_width = (upper - lower)

    buy_signal  = (rsi < strategy_condition.rsi_entry) & (rsi < lower) & bullish & (bb_width < bb_width_threshold)
    sell_signal = (rsi > strategy_condition.rsi_exit)  & (rsi > upper) & bearish & (bb_width < bb_width_threshold)

    # costruisci array di segnali: 1=buy, 0=none, -1=sell
    signals = np.zeros(rsi.shape, np.int64)
    signals[buy_signal]  =  1
    signals[sell_signal] = -1

    return signals