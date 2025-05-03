from numba import njit
import numpy as np
from numba import float32
import polars as pl



@njit
def calculate_max_drawdown_from_initial(equity_curve: np.ndarray, initial_equity: float32) -> float32:
    n = equity_curve.shape[0]
    min_equity = float32(initial_equity)

    for i in range(n):
        v = equity_curve[i]
        if v < min_equity:
            min_equity = v

    return (initial_equity - min_equity) / initial_equity



def compute_strategy_score(df: pl.DataFrame, years: list[int], alpha=0.7, beta=0.2, gamma=0.1) -> pl.DataFrame:
    equity_cols   = [f"equity_{y}"   for y in years]
    drawdown_cols = [f"drawdown_{y}" for y in years]

    # 1) calcolo delle medie e del quadrato medio delle equity
    df = df.with_columns([
        pl.mean_horizontal(pl.col(equity_cols)).alias("mean_equity"),
        pl.mean_horizontal(pl.col(drawdown_cols)).alias("mean_drawdown"),
        pl.mean_horizontal([pl.col(c) ** 2 for c in equity_cols]).alias("mean_squared_equity"),
    ])

    # 2) varianza dell’equity
    df = df.with_columns([
        (pl.col("mean_squared_equity") - pl.col("mean_equity") ** 2).alias("var_equity"),
    ])

    # 3) score finale
    df = df.with_columns([
        ( alpha * pl.col("mean_equity")
        - beta  * pl.col("mean_drawdown")
        - gamma * pl.col("var_equity")
        ).alias("score"),
    ])

    return df






