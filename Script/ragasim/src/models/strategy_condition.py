import numpy as np
from numba import float32, int32
from numba.experimental import jitclass

spec = [
    ('rsi_entry',  float32),
    ('rsi_exit',   float32),
    ('exposure',   float32),
    ('atr_factor', float32),
    ('bb_std',     float32),
    ('atr_window', int32),
]

@jitclass(spec)
class StrategyCondition:
    def __init__(self,
                 rsi_entry: float,
                 rsi_exit: float,
                 exposure: float,
                 atr_factor: float,
                 bb_std: float,
                 atr_window: int):
        # Conversione a float32 per Numba
        self.rsi_entry  = np.float32(rsi_entry)
        self.rsi_exit   = np.float32(rsi_exit)
        self.exposure   = np.float32(exposure)
        self.atr_factor = np.float32(atr_factor)
        # Nuovi parametri
        self.bb_std     = np.float32(bb_std)
        self.atr_window = np.int32(atr_window)