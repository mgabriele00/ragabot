from typing import List
import numpy as np
import os
from model.strategy_indicators import StrategyIndicators, generate_indicators_to_test
from service.signal_service import get_signal
from utils.data_utils import load_forex_data_dohlc, combine_all_years_by_parameters, build_polars_table_for_year
from service.backtesting_service import backtest, backtest_with_trades  # Aggiunto backtest_with_trades
from numba import njit, prange
from model.strategy_condition import StrategyCondition, generate_conditions_to_test
from service.analysis_service import calculate_max_drawdown_from_initial, compute_strategy_score
import polars as pl
from datetime import datetime


@njit(parallel=True, fastmath=True)
def simulate(close:np.ndarray, high:np.ndarray, low:np.ndarray, strategy_indicators:StrategyIndicators, strategy_condition: List[StrategyCondition]):
    n_conditions = len(strategy_condition)
    
    final_equities = np.zeros(n_conditions, dtype=np.float32)
    conditions_indices = np.zeros(n_conditions, dtype=np.int32)
    max_drawdowns = np.zeros(n_conditions, dtype=np.float32)

    
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
        
        equity_curve = backtest(
            close,
            high,
            low,
            strategy_indicators.atr[idx].values,
            signal,
            condition.start_index,
            condition.initial_equity,
            condition.sl_mult,
            condition.tp_mult,
            condition.exposure,
            condition.leverage,
            condition.fixed_fee,
            condition.lot_size
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
    n_conditions = len(strategy_conditions_list)
    print(f"Number of conditions to test: {n_conditions}")
    print(f"Starting simulation for year: {year}...")
    # Ottieni equity_curves e conditions_indices separatamente
    final_equities, condition_indices, max_drawdowns = simulate(close, high, low, strategy_indicators, strategy_conditions_list)
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
    return final_equities, condition_indices, max_drawdowns

def create_trades_dataset(year: int):
    _, open_, high, low, close = load_forex_data_dohlc(year)
    strategy_indicators = generate_indicators_to_test(close, high, low, open_)
    strategy_conditions_list = generate_conditions_to_test()
    
    trades_data = []
    
    # Pattern names - aggiungi qui tutti i pattern che vuoi monitorare
    pattern_names = [
        'doji', 'hammer', 'hanging_man', 'inverted_hammer', 'shooting_star',
        'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star',
        'piercing_pattern', 'dark_cloud_cover', 'three_white_soldiers',
        'three_black_crows', 'bullish_harami', 'bearish_harami',
        'bullish_harami_cross', 'bearish_harami_cross', 'tweezer_top',
        'tweezer_bottom', 'three_inside_up', 'three_inside_down',
        'three_outside_up', 'three_outside_down', 'upside_gap_two_crows',
        'downside_gap_three_methods'
    ]
    
    for condition in strategy_conditions_list:
        signal = get_signal(strategy_indicators, condition, close)
        
        # Trova l'indice ATR corretto
        idx = 0
        for j in range(len(strategy_indicators.atr)):
            if abs(strategy_indicators.atr[j].window - condition.atr_window) < 0.005:
                idx = j
                break
                
        entry_prices, exit_prices, entry_times, exit_times, pnls = backtest_with_trades(
            close, high, low,
            strategy_indicators.atr[idx].values,
            signal,
            strategy_indicators.rsi,
            strategy_indicators.bollinger,
            strategy_indicators.bullish,
            strategy_indicators.bearish,
            condition.start_index,
            condition.initial_equity,
            condition.sl_mult,
            condition.tp_mult,
            condition.exposure,
            condition.leverage,
            condition.fixed_fee,
            condition.lot_size
        )
        
        # Crea record per ogni trade
        for i in range(len(entry_prices)):
            entry_idx = entry_times[i]
            
            # Base trade info
            trade = {
                'entry_price': entry_prices[i],
                'exit_price': exit_prices[i],
                'entry_time': entry_times[i],
                'exit_time': exit_times[i],
                'pnl': pnls[i],
                'target': 1 if pnls[i] > 0 else 0,
                'trade_duration': exit_times[i] - entry_times[i],
                
                # Strategy parameters
                'rsi_entry': condition.rsi_entry,
                'rsi_exit': condition.rsi_exit,
                'bb_width': condition.bb_width_threshold,
                'bb_std': condition.bb_std,
                'sl_mult': condition.sl_mult,
                'tp_mult': condition.tp_mult,
                'atr_window': condition.atr_window,
                
                # Market indicators at entry
                'rsi_value': strategy_indicators.rsi[entry_idx],
                'bb_upper': strategy_indicators.bollinger.upper[entry_idx],
                'bb_lower': strategy_indicators.bollinger.lower[entry_idx],
                'bb_middle': strategy_indicators.bollinger.middle[entry_idx],
                'atr_value': strategy_indicators.atr[idx].values[entry_idx],
                
                # Price action at entry
                'open': open_[entry_idx],
                'high': high[entry_idx],
                'low': low[entry_idx],
                'close': close[entry_idx],
                'body_size': abs(close[entry_idx] - open_[entry_idx]),
                'upper_shadow': high[entry_idx] - max(open_[entry_idx], close[entry_idx]),
                'lower_shadow': min(open_[entry_idx], close[entry_idx]) - low[entry_idx],
                'candle_range': high[entry_idx] - low[entry_idx],
            }
            
            # Aggiungi tutti i pattern come colonne separate
            for pattern in pattern_names:
                pattern_value = getattr(strategy_indicators, pattern)[entry_idx] if hasattr(strategy_indicators, pattern) else 0
                trade[f'pattern_{pattern}'] = pattern_value
            
            trades_data.append(trade)
    
    # Crea DataFrame Polars
    df = pl.DataFrame(trades_data)
    
    # Calcola alcune statistiche aggiuntive
    if len(df) > 0:
        df = df.with_columns([
            (pl.col('body_size') / pl.col('candle_range')).alias('body_to_range_ratio'),
            ((pl.col('upper_shadow') + pl.col('lower_shadow')) / pl.col('candle_range')).alias('shadow_to_range_ratio')
        ])
    
    # Salva il CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.write_csv(f"results/trades_dataset_{year}_{timestamp}.csv")
    return df

if __name__ == '__main__':
    # Scegli quale modalità eseguire
    mode = "backtest"  # oppure "backtest"
    
    if mode == "dataset":
        # Genera dataset per 2013-2014
        df_2013 = create_trades_dataset(2013)
        df_2014 = create_trades_dataset(2014)
        df_2015 = create_trades_dataset(2015)
        df_2016 = create_trades_dataset(2016)
        df_2017 = create_trades_dataset(2017)
        df_2018 = create_trades_dataset(2018)
        df_2019 = create_trades_dataset(2019)
        df_2020 = create_trades_dataset(2020)
        df_2021 = create_trades_dataset(2021)
        df_2022 = create_trades_dataset(2022)
        df_2023 = create_trades_dataset(2023)
        df_2024 = create_trades_dataset(2024)
        # Combina i dataset
        df_combined = pl.concat([df_2013, df_2014,df_2015,df_2016,df_2017,df_2018,df_2019,df_2020,df_2021,df_2022,df_2023,df_2024])
        
        # Salva il dataset combinato
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_combined.write_csv(f"results/trades_dataset_combined_{timestamp}.csv")
        print("Dataset creato con successo!")
    else:
        years = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
        os.makedirs("results", exist_ok=True)
        
        # Combina tutto usando la funzione esterna
        df_all = combine_all_years_by_parameters(years, main)
        result_df = compute_strategy_score(df_all, years)
        top500 = (result_df.sort("score",descending=True).head(500))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        top500.write_csv(f"results/top500_{timestamp}.csv")
        print("CSV salvato")
