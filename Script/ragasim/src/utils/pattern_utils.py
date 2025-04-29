import numpy as np
import talib

def get_pattern(close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bullish, bearish = np.zeros(len(close), bool), np.zeros(len(close), bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        result = getattr(talib, pattern)(open_, high, low, close)
        bullish |= result > 0
        bearish |= result < 0
    return bullish, bearish    