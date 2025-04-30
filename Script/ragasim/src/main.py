#main
import numpy as np
import csv
from models.strategy_indicators import generate_indicators_to_test
from service.param_simulation_service import get_signal
from utils.data_utils import load_forex_data_dohlc, save_results_to_csv # Rimuovi save_results_to_csv da qui se definita sotto
from service.backtesting_service import backtest
from numba import njit, prange
# Importa la classe StrategyCondition per accedere ai suoi campi nel salvataggio
from models.strategy_condition import generate_conditions_to_test

# Attiva parallel=True e usa prange
@njit(parallel=True, fastmath=True)
def simulate(close, strategy_indicators, strategy_condition):
    n_conditions = len(strategy_condition)
    # Array per i risultati: equity, pnl, md, win_rate (4 colonne)
    results = np.empty((n_conditions, 4), dtype=np.float64)

    for i in prange(n_conditions):
        condition = strategy_condition[i]
        signal = get_signal(strategy_indicators, condition)
        # Ricevi i 4 valori da backtest
        final_equity, final_pnl, final_md, final_wr = backtest(signal, close, strategy_indicators, condition, 1000.0, 100.0, condition.exposure)

        results[i, 0] = final_equity
        results[i, 1] = final_pnl
        results[i, 2] = final_md
        results[i, 3] = final_wr # Salva il Win Rate

    return results

def main(year: int):
    datetime, open_, high, low, close = load_forex_data_dohlc(year)
    # Rinomina le variabili per chiarezza (params vs conditions)
    strategy_indicators = generate_indicators_to_test(close, high, low, open_)
    strategy_conditions_list = generate_conditions_to_test()
    n_conditions = len(strategy_conditions_list)
    print(f"Number of conditions to test: {n_conditions}")

    print(f"Starting simulation for year: {year}...")
    # Passa la lista di condizioni a simulate
    simulation_results = simulate(close, strategy_indicators, strategy_conditions_list)
    print("Simulation finished.")

    output_filename = f"simulation_results_{year}.csv"
    # Passa sia i risultati che la lista delle condizioni alla funzione di salvataggio
    save_results_to_csv(simulation_results, strategy_conditions_list, output_filename)

if __name__ == '__main__':
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    for year in years:
        main(year)

