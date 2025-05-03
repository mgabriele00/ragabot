import numpy as np
from numba import float32, int32
from numba.experimental import jitclass
import itertools # Importa itertools
from typing import List # Per il type hint

spec = [
    ('rsi_entry',  float32),
    ('rsi_exit',   float32),
    ('exposure',   float32),
    ('atr_factor', float32),
    ('bb_std',     float32),
    ('atr_window', int32),
    ('bb_width_threshold', float32), # Aggiunto nuovo parametro
]

@jitclass(spec)
class StrategyCondition:
    def __init__(self,
                 rsi_entry: float,
                 rsi_exit: float,
                 exposure: float,
                 atr_factor: float,
                 bb_std: float,
                 atr_window: int,
                 bb_width_threshold: float): # Aggiunto nuovo parametro
        # Conversione a float32 per Numba
        self.rsi_entry  = np.float32(rsi_entry)
        self.rsi_exit   = np.float32(rsi_exit)
        self.exposure   = np.float32(exposure)
        self.atr_factor = np.float32(atr_factor)
        self.bb_std     = np.float32(bb_std)
        self.atr_window = np.int32(atr_window)
        self.bb_width_threshold = np.float32(bb_width_threshold) # Aggiunto nuovo parametro

# --- Funzione definita FUORI dalla classe ---

def generate_conditions_to_test() -> List[StrategyCondition]:
    """
    Genera una lista di tutte le combinazioni possibili di StrategyCondition
    basate su intervalli di parametri predefiniti.

    Returns:
        List[StrategyCondition]: Una lista di oggetti StrategyCondition.
    """
    params = {
        "rsi_entry": list(range(20, 45)),
        "rsi_exit": list(range(55, 80)),
        "bb_std": [1.75, 2, 2.25],
        "exposure": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "atr_window": [14, 20,9,27],
        "atr_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        "bb_width_threshold": [0.001, 0.008, 0.0015, 0.002]
    }

    param_values_ordered = [
        params["rsi_entry"],
        params["rsi_exit"],
        params["exposure"],
        params["atr_factor"],
        params["bb_std"],
        params["atr_window"],
        params["bb_width_threshold"] 
    ]
    param_combinations = list(itertools.product(*param_values_ordered))
    conditions = [StrategyCondition(*combo) for combo in param_combinations]
    return conditions


