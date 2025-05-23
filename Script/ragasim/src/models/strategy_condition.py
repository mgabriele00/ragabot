import numpy as np
from numba import float32, int32
from numba.experimental import jitclass
import itertools # Importa itertools
from typing import List # Per il type hint

spec = [
    ('rsi_entry',  float32),
    ('rsi_exit',   float32),
    ('exposure',   float32),
    ('tp_mult',    float32),
    ('sl_mult',    float32),
    ('bb_std',     float32),
    ('atr_window', int32),
    ('bb_width_threshold', float32), 
]

@jitclass(spec)
class StrategyCondition:
    def __init__(self,
                 rsi_entry: float,
                 rsi_exit: float,
                 exposure: float,
                 tp_mult: float,
                 sl_mult: float,
                 bb_std: float,
                 atr_window: int,
                 bb_width_threshold: float): 
        # Conversione a float32 per Numba
        self.rsi_entry  = np.float32(rsi_entry)
        self.rsi_exit   = np.float32(rsi_exit)
        self.exposure   = np.float32(exposure)
        self.bb_std     = np.float32(bb_std)
        self.atr_window = np.int32(atr_window)
        self.tp_mult    = np.float32(tp_mult)
        self.sl_mult    = np.float32(sl_mult)
        self.bb_width_threshold = np.float32(bb_width_threshold) 

# --- Funzione definita FUORI dalla classe ---

def generate_conditions_to_test() -> List[StrategyCondition]:
    """
    Genera una lista di tutte le combinazioni possibili di StrategyCondition
    basate su intervalli di parametri predefiniti.

    Returns:
        List[StrategyCondition]: Una lista di oggetti StrategyCondition.
    """
    params = {
<<<<<<< Updated upstream
        "rsi_entry": [35],
        "rsi_exit": [55],
        "bb_std": [1.75],
        "exposure": [0.6],
        "atr_window": [14],
        "tp_mult": [0.02],
        "sl_mult": [0.006],
        "bb_width_threshold": [0.002]
=======
        "rsi_entry": [30,40,35],
        "rsi_exit": [55,60,65,70],
        "bb_std": [1.75, 2, 2.25],
        "exposure": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "atr_window": [14, 20],
        "atr_factor": [1.0,  2.0, 5, 10],
        "bb_width_threshold": [0.001, 0.002]
>>>>>>> Stashed changes
    }

    param_values_ordered = [
        params["rsi_entry"],
        params["rsi_exit"],
        params["exposure"],
        params["tp_mult"],
        params["sl_mult"],
        params["bb_std"],
        params["atr_window"],
        params["bb_width_threshold"] 
    ]
    param_combinations = list(itertools.product(*param_values_ordered))
    conditions = [StrategyCondition(*combo) for combo in param_combinations]
    return conditions


