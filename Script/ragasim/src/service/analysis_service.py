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



import polars as pl

def compute_strategy_score(
    df: pl.DataFrame,
    years: list[int],
    alpha=0.7,
    beta=0.2,
    gamma=0.1
) -> pl.DataFrame:
    equity_cols   = [f"equity_{y}"   for y in years]
    drawdown_cols = [f"drawdown_{y}" for y in years]

    # 1) medie e media dei quadrati
    df = df.with_columns([
        pl.mean_horizontal(pl.col(equity_cols)).alias("mean_equity"),
        pl.mean_horizontal(pl.col(drawdown_cols)).alias("mean_drawdown"),
        pl.mean_horizontal([pl.col(c) ** 2 for c in equity_cols]).alias("mean_squared_equity"),
    ])

    # 2) varianza e deviazione standard
    df = df.with_columns([
        # varianza = E[X^2] - (E[X])^2
        (pl.col("mean_squared_equity") - pl.col("mean_equity")**2).alias("var_equity"),
        # deviazione standard
        pl.sqrt(pl.col("mean_squared_equity") - pl.col("mean_equity")**2).alias("std_equity"),
    ])

    # 3) volatilità relativa = std_equity / mean_equity
    df = df.with_columns([
        (pl.col("std_equity") / pl.col("mean_equity")).alias("rel_volatility")
    ])

    # 4) score = α·mean_equity - β·mean_drawdown - γ·rel_volatility
    df = df.with_columns([
        (
            alpha * pl.col("mean_equity")
          - beta  * pl.col("mean_drawdown")
          - gamma * pl.col("rel_volatility")
        ).alias("score")
    ])

    return df







