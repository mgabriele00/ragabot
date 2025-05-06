from numba import njit
import numpy as np
from numba import float32
import polars as pl



@njit
def calculate_max_drawdown_from_initial(equity_curve: np.ndarray, initial_equity: float32) -> float32:
    n = equity_curve.shape[0]
    min_equity = float32(initial_equity)
    for v in equity_curve:
        # Ignoro i valori zero
        if v <= 0:
            continue
        # Se trovo un valore > 0 ma minore del minimo corrente, aggiorno
        if v < min_equity:
            min_equity = v
    
    return (initial_equity - min_equity)/ initial_equity



def compute_strategy_score(
    df: pl.DataFrame,
    years: list[int],
    initial: float = 1_000.0,
    alpha: float = 0.7,
    beta:  float = 0.2,
    gamma: float = 0.1
) -> pl.DataFrame:
    # 1) Definisco i nomi delle colonne di rendimento e drawdown (già in 0–1)
    ret_cols = [f"ret_{y}" for y in years]
    dd_cols  = [f"dd_{y}"  for y in years]
    
    # 2) Aggiungo le colonne di rendimento percentuale e drawdown
    df = df.with_columns([
        # rendimento: equity_y / initial - 1
        ((pl.col(f"equity_{y}") / initial) - 1).alias(f"ret_{y}")
        for y in years
    ] + [
        # drawdown_y: prendo direttamente il valore in 0–1
        pl.col(f"drawdown_{y}").alias(f"dd_{y}")
        for y in years
    ])
    
    # 3a) Calcolo media dei rendimenti, media dei quadrati e media dei drawdown
    df = df.with_columns([
        pl.mean_horizontal(pl.col(ret_cols)).alias("mean_return"),
        pl.mean_horizontal([pl.col(c)**2 for c in ret_cols]).alias("mean_sq_return"),
        pl.mean_horizontal(pl.col(dd_cols)).alias("mean_drawdown"),
    ])
    
    # 3b) Varianza e deviazione standard assoluta dei rendimenti
    df = df.with_columns([
        # var_return = E[R^2] - (E[R])^2
        (pl.col("mean_sq_return") - pl.col("mean_return")**2).alias("var_return"),
        # std_return = sqrt(var_return)
        (pl.col("mean_sq_return") - pl.col("mean_return")**2).sqrt().alias("std_return"),
    ])
    
    # 4) Coefficiente di variazione (volatilità relativa, unitless)
    df = df.with_columns([
        (pl.col("std_return") / pl.col("mean_return")).alias("rel_volatility")
    ])
    
    # 5) Calcolo dello score finale
    df = df.with_columns([
        (
            alpha * pl.col("mean_return")
          - beta  * pl.col("mean_drawdown")
          - gamma * pl.col("rel_volatility")
        ).alias("score")
    ])
    
    return df






