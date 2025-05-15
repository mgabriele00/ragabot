from typing import List
import numpy as np
import os
from model.strategy_indicators import StrategyIndicators, generate_indicators_to_test
from service.signal_service import get_signal
from utils.data_utils import load_forex_data_dohlc, combine_all_years_by_parameters, build_polars_table_for_year
from service.backtesting_service import backtest
from numba import njit, prange
from model.strategy_condition import StrategyCondition, generate_conditions_to_test
from service.analysis_service import calculate_max_drawdown_from_initial, compute_strategy_score
import polars as pl
from datetime import datetime

@njit(parallel=True, fastmath=True)
def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             sigma: np.ndarray, atr: List[np.ndarray],
             strategy_conditions: List[StrategyCondition],
             strategy_indicators: StrategyIndicators) -> tuple:
    n_conditions = len(strategy_conditions)
    final_equities = np.zeros(n_conditions, dtype=np.float32)
    conditions_indices = np.zeros(n_conditions, dtype=np.int32)
    max_drawdowns = np.zeros(n_conditions, dtype=np.float32)
    
    # Definisci n_sim e mc_threshold come costanti
    N_SIM = 10000  
    MC_THRESHOLD = 0.7  # Valore predefinito per mc_threshold

    for i in prange(n_conditions):
        condition = strategy_conditions[i]
        signal = get_signal(strategy_indicators, condition, close)

        idx = 0
        tol = 0.005
        for j in range(len(atr)):
            if abs(atr[j].window - condition.atr_window) < tol:
                idx = j
                break

        equity_curve = backtest(
            close,
            high,
            low,
            atr[idx].values,
            sigma,
            signal,
            condition.start_index,
            condition.initial_equity,
            condition.sl_mult,
            condition.tp_mult,
            condition.exposure,
            condition.leverage,
            condition.fixed_fee,
            condition.lot_size,
            N_SIM,
            MC_THRESHOLD  # Usa il valore costante invece di condition.mc_threshold
        )

        final_equity = equity_curve[-1]
        final_equities[i] = final_equity
        conditions_indices[i] = i
        max_drawdowns[i] = calculate_max_drawdown_from_initial(equity_curve, condition.initial_equity)

    return final_equities, conditions_indices, max_drawdowns

def main(year: int):
    _, open_, high, low, close = load_forex_data_dohlc(year)
    strategy_indicators = generate_indicators_to_test(close, high, low, open_)
    strategy_conditions_list = generate_conditions_to_test()

    print(f"Number of conditions to test: {len(strategy_conditions_list)}")
    print(f"Starting simulation for year: {year}...")

    sigma = np.zeros_like(close)
    log_ret = np.log(close[1:] / close[:-1])
    for i in range(100, len(close)):
        sigma[i] = np.std(log_ret[i-100:i])

    final_equities, condition_indices, max_drawdowns = simulate(
        close, high, low, sigma, strategy_indicators.atr, 
        strategy_conditions_list, strategy_indicators  # Aggiunto parametro
    )

    print("Simulation finished.")
    print("Max equity:", np.max(final_equities))
    best_idx = np.argmax(final_equities)
    print("Drawdown della strategia migliore:", max_drawdowns[best_idx])
    condition = strategy_conditions_list[condition_indices[best_idx]]

    print("RSI Entry:", condition.rsi_entry)
    print("RSI Exit:", condition.rsi_exit)
    print("BB Width:", condition.bb_width_threshold)
    print("BB Std:", condition.bb_std)
    print("SL Mult:", condition.sl_mult)
    print("TP Mult:", condition.tp_mult)
    print("Exposure:", condition.exposure)
    print("Atr Window:", condition.atr_window)
    print("Max dd:", max_drawdowns[best_idx])
    print("MC Threshold:", condition.mc_threshold)
    print("N Sim:", condition.n_sim)
    return final_equities, condition_indices, max_drawdowns

if __name__ == '__main__':
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    os.makedirs("results", exist_ok=True)
    df_all = combine_all_years_by_parameters(years, main)
    result_df = compute_strategy_score(df_all, years)
    top500 = result_df.sort("score", descending=True).head(500)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top500.write_csv(f"Script/ragasim/app/results/top500_{timestamp}.csv")
    print("CSV salvato")