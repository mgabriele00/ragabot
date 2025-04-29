from typing import List
import talib
import numpy as np

from models.strategy_condition import StrategyCondition
from Script.ragasim.src.models.strategy_indicators import StrategyIndicators
from utils.pattern_utils import get_pattern
from numba import njit

def generate_indicators_to_test(close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray) -> StrategyIndicators:
    rsi = talib.RSI(close, timeperiod=14)
    bullish, bearish = get_pattern(close, open_, high, low)
    
    bb_std = [1.5, 1.75, 2.0]
    atr_window = [14, 20]
    
    bollinger_bands = []
    for std in bb_std:
        upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=std, nbdevdn=std)
        bollinger_bands.append((std, lower, middle, upper))
    atr_data = []
    for window in atr_window:
        atr = talib.ATR(high, low, close, timeperiod=window)
        atr_data.append((window, atr))
                
    return StrategyIndicators(rsi, bollinger_bands, atr_data, bullish, bearish)

def generate_conditions_to_test(params: StrategyIndicators) -> List[StrategyCondition]:
  params = {
    "rsi_entry": list(range(30, 46)),
    "rsi_exit": list(range(55, 71)),
    "bb_std": [1.5, 1.75, 2.0],
    "exposure": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "atr_window": [14, 20],
    "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    }
    
  conditions = []
  for rsi_entry in params["rsi_entry"]:
        for rsi_exit in params["rsi_exit"]:
            for bb_std in params["bb_std"]:
                for exposure in params["exposure"]:
                    for atr_window in params["atr_window"]:
                        for atr_factor in params["atr_factor"]:
                            conditions.append(StrategyCondition(rsi_entry, rsi_exit, exposure, atr_factor, bb_std, atr_window))
                            
  return conditions

@njit
def get_signal(strategy_params, strategy_condition) -> np.ndarray:
    rsi      = strategy_params.rsi
    bullish  = strategy_params.bullish
    bearish  = strategy_params.bearish
    bb_std   = strategy_condition.bb_std
    # trova l’indice nella lista bollinger corrispondente a questo bb_std
    idx = 0
    for i, bb in enumerate(strategy_params.bollinger):
        if bb.bb_std[0] == bb_std:
            idx = i
            break
    upper = strategy_params.bollinger[idx].upper
    lower = strategy_params.bollinger[idx].lower

    buy_signal  = (rsi < strategy_condition.rsi_entry) & (rsi < lower) & bullish
    sell_signal = (rsi > strategy_condition.rsi_exit)  & (rsi > upper) & bearish

    # costruisci array di segnali: 1=buy, 0=none, -1=sell
    signals = np.zeros(rsi.shape, np.int64)
    signals[buy_signal]  =  1
    signals[sell_signal] = -1

    return signals


