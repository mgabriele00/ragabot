import numpy as np
import talib
from numba import float32, int32, boolean, types
from numba.typed import List
from numba.experimental import jitclass

from Script.ragasim.src.utils.pattern_utils import get_pattern

# 1) BollingerBand
boll_spec = [
    ('bb_std',  float32),
    ('lower',   float32[:]),
    ('medium',  float32[:]),
    ('upper',   float32[:]),
]
@jitclass(boll_spec)
class BollingerBand:
    def __init__(self, bb_std, lower, medium, upper):
        self.bb_std = bb_std
        self.lower  = lower
        self.medium = medium
        self.upper  = upper

# 2) ATRParams
atr_spec = [
    ('window', int32),
    ('values', float32[:]),
]
@jitclass(atr_spec)
class ATRParams:
    def __init__(self, window, values):
        self.window = window
        self.values = values

# Estraiamo qui i type-objects da usare in spec e in __init__
BB_T = BollingerBand.class_type.instance_type
ATR_T = ATRParams.class_type.instance_type

# 3) StrategyIndicators
strat_spec = [
    ('rsi',       float32[:]),
    ('bollinger', types.ListType(BB_T)),
    ('atr',       types.ListType(ATR_T)),
    ('bullish',   boolean[:]),
    ('bearish',   boolean[:]),
]

@jitclass(strat_spec)
class StrategyIndicators:
    def __init__(self, rsi, boll_data, atr_data, bullish, bearish):
        # RSI
        self.rsi = rsi.astype(np.float32)

        # Bollinger
        bb_list = List.empty_list(BB_T)
        for bb_std, lower, medium, upper in boll_data:
            # bb_std può essere uno scalar float32: creiamo un array float32
            bb_arr = np.full(lower.shape, np.float32(bb_std))
            bb_list.append(
                BollingerBand(
                    bb_arr,
                    lower.astype(np.float32),
                    medium.astype(np.float32),
                    upper.astype(np.float32),
                )
            )
        self.bollinger = bb_list

        # ATR
        atr_list = List.empty_list(ATR_T)
        for window, values in atr_data:
            atr_list.append(
                ATRParams(
                    np.int32(window),
                    values.astype(np.float32),
                )
            )
        self.atr = atr_list

        # Boolean arrays
        self.bullish = bullish.astype(np.bool_)
        self.bearish = bearish.astype(np.bool_)

# --- Funzione definita FUORI dalla classe ---

def generate_indicators_to_test(close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray) -> StrategyIndicators:
    """
    Calcola gli indicatori tecnici necessari e crea un'istanza di StrategyIndicators.

    Args:
        close (np.ndarray): Array dei prezzi di chiusura.
        high (np.ndarray): Array dei prezzi massimi.
        low (np.ndarray): Array dei prezzi minimi.
        open_ (np.ndarray): Array dei prezzi di apertura.

    Returns:
        StrategyIndicators: Un'istanza contenente tutti gli indicatori calcolati.
    """
    # Calcola RSI
    rsi = talib.RSI(close, timeperiod=14)

    # Calcola Pattern Bullish/Bearish
    bullish, bearish = get_pattern(close, open_, high, low)

    # Definisci i parametri da testare per BB e ATR
    bb_std_values = [1.5, 1.75, 2.0]
    atr_window_values = [14, 20]

    # Calcola Bollinger Bands per ogni deviazione standard
    bollinger_bands_data = []
    for std in bb_std_values:
        # Assicurati che timeperiod sia abbastanza piccolo rispetto alla lunghezza di 'close'
        # e che 'close' non contenga troppi NaN all'inizio.
        try:
            upper, middle, lower = talib.BBANDS(close, timeperiod=14, nbdevup=std, nbdevdn=std)
            # Aggiungi la tupla (std, lower, middle, upper)
            bollinger_bands_data.append((std, lower, middle, upper))
        except Exception as e:
            print(f"Errore nel calcolo di BBANDS con std={std}: {e}")
            # Potresti voler gestire l'errore in modo diverso, es. saltando questo std
            # o restituendo un valore di errore/None

    # Calcola ATR per ogni finestra
    atr_data_list = []
    for window in atr_window_values:
        # Assicurati che high, low, close non contengano troppi NaN all'inizio
        try:
            atr = talib.ATR(high, low, close, timeperiod=window)
            # Aggiungi la tupla (window, atr_values)
            atr_data_list.append((window, atr))
        except Exception as e:
            print(f"Errore nel calcolo di ATR con window={window}: {e}")
    return StrategyIndicators(rsi, bollinger_bands_data, atr_data_list, bullish, bearish)