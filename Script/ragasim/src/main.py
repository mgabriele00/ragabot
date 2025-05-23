#main
from typing import List
import numpy as np
import os
from models.strategy_indicators import StrategyIndicators, generate_indicators_to_test
from service.signal_service import get_signal
from utils.data_utils import load_forex_data_dohlc, save_results_to_csv, combine_all_years_by_parameters, build_polars_table_for_year # Rimuovi save_results_to_csv da qui se definita sotto
from service.backtesting_service import backtest
from numba import njit, prange
from models.strategy_condition import StrategyCondition, generate_conditions_to_test
from service.analysis_service import compute_strategy_score



@njit(parallel=True, fastmath=False)
def simulate(close:np.ndarray, strategy_indicators:StrategyIndicators, strategy_condition: List[StrategyCondition]):
    n_conditions = len(strategy_condition)
    
    final_equities = np.zeros(n_conditions, dtype=np.float32)
    conditions_indices = np.zeros(n_conditions, dtype=np.int32)
    max_drawdowns = np.zeros(n_conditions, dtype=np.float32)
    trade_counts = np.zeros(n_conditions, dtype=np.float32)
    gross_profits = np.zeros(n_conditions, dtype=np.float32)
    gross_losss = np.zeros(n_conditions, dtype=np.float32)
    avg_pnls = np.zeros(n_conditions, dtype=np.float32)
    total_feess = np.zeros(n_conditions, dtype=np.float32)
    fee_ratios = np.zeros(n_conditions, dtype=np.float32)
    
    for i in prange(n_conditions):
        condition = strategy_condition[i]
        signal = get_signal(strategy_indicators, condition, close)
        
        idx = 0
        tol = 0.005 
        for j in range(len(strategy_indicators.atr)): 
            atr_window = strategy_indicators.atr[j].window
            if abs(atr_window - condition.atr_window) < tol:
                idx = j
                break
        
        final_equity, max_dd, trade_count, gross_profit, gross_loss, avg_pnl, total_fees, fee_ratio  = backtest(
            close,
            strategy_indicators.atr[idx].values,
            signal,
            1000,
            condition.sl_mult,
            condition.tp_mult,
            condition.exposure,
            30
        )
        
        final_equities[i] = final_equity
        conditions_indices[i] = i
        max_drawdowns[i] = max_dd
        trade_counts[i] = trade_count
        gross_profits[i] =gross_profit
        gross_losss[i] = gross_loss
        avg_pnls[i] = avg_pnl
        total_feess[i] = total_fees
        fee_ratios[i] = fee_ratio

    return final_equities, conditions_indices, max_drawdowns, trade_counts, gross_profits, gross_losss, avg_pnls, total_feess, fee_ratios

def main(year: int):
    _, open_, high, low, close = load_forex_data_dohlc(year)
    strategy_indicators = generate_indicators_to_test(close, high, low, open_)
    strategy_conditions_list = generate_conditions_to_test()
    n_conditions = len(strategy_conditions_list)
    print(f"Number of conditions to test: {n_conditions}")
    print(f"Starting simulation for year: {year}...")
    # Ottieni equity_curves e conditions_indices separatamente
    final_equities, condition_indices, max_drawdowns, trade_counts, gross_profits, gross_losss, avg_pnls, total_feess, fee_ratios = simulate(close, strategy_indicators, strategy_conditions_list)
    print("Simulation finished.")
    print("Max equity:", np.max(final_equities))
    best_idx = np.argmax(final_equities)
    print("Drawdown della strategia migliore:", max_drawdowns[best_idx])
    condition_index = condition_indices[np.argmax(final_equities)]
    condition = strategy_conditions_list[condition_index]
    print("RSI Entry: ", condition.rsi_entry)
    print("RSI Exit: ", condition.rsi_exit)
    print("BB Width: ", condition.bb_width_threshold)
    print("BB Std: ", condition.bb_std)
    print("SL Mult: ", condition.sl_mult)
    print("TP Mult: ", condition.tp_mult)
    print("Exposure: ", condition.exposure)
    print("Atr Window: ", condition.atr_window)
    print("Max dd:", np.max(max_drawdowns))
    print("Trade counts", trade_counts[best_idx])
    print("Gross Profit", gross_profits[best_idx])
    print("Gross Loss", gross_losss[best_idx])
    print("Avg PLN", avg_pnls[best_idx])
    print("Total Fees", total_feess[best_idx])
    print("fee ratios", fee_ratios[best_idx])
    return final_equities, condition_indices, max_drawdowns, trade_counts, gross_profits, gross_losss, avg_pnls, total_feess, fee_ratios

if __name__ == '__main__':
<<<<<<< Updated upstream
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    #years = [2013]
=======
    years = [2013]

>>>>>>> Stashed changes
    os.makedirs("results", exist_ok=True)

    # Combina tutto usando la funzione esterna
    df_all = combine_all_years_by_parameters(years, main)
    df_scored = compute_strategy_score(df_all, years, alpha=0.5, beta=0.3, gamma=0.2)
    df_scored = df_scored.sort("score", descending=True)
    df_scored.write_csv("results/scored_strategies_by_params.csv")
    print("✅ File salvato in results/strategies_by_params_all_years.csv")
