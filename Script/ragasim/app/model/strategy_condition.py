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
    ('fixed_fee', float32),
    ('initial_equity', float32),
    ('leverage', int32),
    ('lot_size', int32),
    ('waiting_number', int32),
    ('start_index', int32),
]

"""
params = {
    "rsi_entry": [40, 41, 42, 43, 44, 45],
    "rsi_exit": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    "bb_std": [1.3, 1.4, 1.5, 1.6, 1.75],
    "exposure": [0.5, 0.6,0.7,0.8,0.9],
    "atr_window": [14],
    "tp_mult": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "sl_mult": [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    "bb_width_threshold": [0.001,0.002,0.003,0.004,0.005,0.006,0.008, 0.01],
    "fixed_fee": [2],
    "initial_equity": [1000],
    "leverage": [30],
    "lot_size": [100000],
    "start_index": [14]
}

"""

params = {
    "rsi_entry": [30, 35, 40, 45],
    "rsi_exit": [55, 60, 65, 70, 75],
    "bb_std": [1.5, 1.3, 1.75],
    "exposure": [0.9],
    "atr_window": [14],
    "tp_mult": [0.5, 0.1, 0.3],
    "sl_mult": [2, 3, 6, 7, 8],
    "bb_width_threshold": [0.001, 0.002, 0.003],
    "leverage": [30],
    "lot_size": [100000],
    "start_index": [14],
    "fixed_fee": [2.1],
    "initial_equity": [1000],
    "waiting_number": [1, 3, 5, 7, 10, 15, 18, 20],
}

param_values_ordered = [
    params["rsi_entry"],
    params["rsi_exit"],
    params["exposure"],
    params["tp_mult"],
    params["sl_mult"],
    params["bb_std"],
    params["atr_window"],
    params["bb_width_threshold"],
    params["fixed_fee"],
    params["initial_equity"],
    params["leverage"],
    params["lot_size"],
    params["waiting_number"],
    params["start_index"],
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
                 bb_width_threshold: float,
                 fixed_fee: float,
                 initial_equity: float,
                 leverage: int,
                 lot_size: int,
                 waiting_number: int,
                 start_index: int = 14,
                 ):
        # Conversione a float32 per Numba
        self.rsi_entry  = np.float32(rsi_entry)
        self.rsi_exit   = np.float32(rsi_exit)
        self.exposure   = np.float32(exposure)
        self.bb_std     = np.float32(bb_std)
        self.atr_window = np.int32(atr_window)
        self.tp_mult    = np.float32(tp_mult)
        self.sl_mult    = np.float32(sl_mult)
        self.bb_width_threshold = np.float32(bb_width_threshold)
        self.fixed_fee = np.float32(fixed_fee)
        self.initial_equity = np.float32(initial_equity)
        self.leverage = np.int32(leverage)
        self.lot_size = np.int32(lot_size)
        self.waiting_number = np.int32(waiting_number)
        self.start_index = np.int32(start_index)

def generate_conditions_to_test() -> List[StrategyCondition]:
    param_combinations = list(itertools.product(*param_values_ordered))
    conditions = [StrategyCondition(*combo) for combo in param_combinations]
    return conditions


