#main
import numpy as np
import csv  # Importa il modulo csv
from service.param_simulation_service import generate_conditions_to_test, generate_indicators_to_test, get_signal
from utils.data_utils import load_data
from service.backtesting_service import backtest
from numba import njit, prange # Importa prange per il loop parallelo

# Rimuovi parallel=True per ora se ci sono problemi con la raccolta dei risultati
# @njit(parallel=True)
@njit(fastmath=True)
def simulate(close, strategy_params, strategy_condition):
    n_conditions = len(strategy_condition)
    # Pre-alloca array per i risultati
    indices = np.empty(n_conditions, dtype=np.int64)
    results = np.empty(n_conditions, dtype=np.float64)

    # Usa prange se parallel=True è attivo, altrimenti range
    # for i in prange(n_conditions): # Usa prange per il loop parallelo
    for i in range(n_conditions): # Usa range se non parallelo
        condition = strategy_condition[i]
        # Rimuovi la stampa da dentro la funzione Numba
        # print(f"Testing condition {i}")
        signal = get_signal(strategy_params, condition)
        cash = backtest(signal, close, strategy_params, condition, 1000.0, 100.0, condition.exposure) # Usa float per initial_cash e leverage
        indices[i] = i
        if(i % 1000 == 0):
            print(f"Testing condition {i}")
        results[i] = cash
    return indices, results # Restituisci entrambi gli array

if __name__ == '__main__':
    date, time, open_, high, low, close = load_data("../../../dati_forex/EURUSD/DAT_MT_EURUSD_M1_2013.csv")

    strategy_params = generate_indicators_to_test(close, high, low, open_)
    strategy_condition = generate_conditions_to_test(strategy_params)
    print(f"Number of conditions to test: {len(strategy_condition)}")

    print("Starting simulation...")
    indices, final_cash_values = simulate(close, strategy_params, strategy_condition)
    print("Simulation finished.")

    # Salva i risultati in un file CSV
    output_filename = "simulation_results.csv"
    print(f"Saving results to {output_filename}...")
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Scrivi l'intestazione
        writer.writerow(['ConditionIndex', 'FinalCash'])
        # Scrivi i dati
        for i in range(len(indices)):
            writer.writerow([indices[i], final_cash_values[i]])

    print("Results saved.")