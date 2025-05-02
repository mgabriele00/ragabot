from numba import njit
import numpy as np
from numba import float32

@njit
def calculate_max_drawdown_from_initial(equity_curve: np.ndarray, initial_equity: float) -> float:
    min_equity = initial_equity
    for i in range(len(equity_curve)):
        value = equity_curve[i]
        if value < min_equity:
            min_equity = value
    drawdown = (initial_equity - min_equity) / initial_equity
    return drawdown







