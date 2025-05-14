import vectorbtpro as vbt
import pandas as pd
import numpy as np
from numba import njit
import talib
from vectorbtpro import *

# Costanti
DATA_FILE = './dati_forex/EURUSD/DAT_MT_EURUSD_M1_2014.csv'
# Parametri
exposure_pct = 0.9
tp_mult = 1000
sl_mult = 1000
expiration = 1000
bb_width_threshold = 0.003
rsi_entry = 50
rsi_exit  = 55
bb_std = 1
init_cash = 1000.0

def get_pattern(close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bullish, bearish = np.zeros(len(close), bool), np.zeros(len(close), bool)
    for pattern in talib.get_function_groups()["Pattern Recognition"]:
        result = getattr(talib, pattern)(open_, high, low, close)
        bullish |= result > 0
        bearish |= result < 0
    return bullish, bearish
def clean_signal(entry: np.ndarray, exit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean the entry and exit signals to avoid open without close."""
    # Creo una copia per non modificare gli array originali
    clean_entry = entry.copy()
    clean_exit = exit.copy()
    
    # Variabile per tracciare lo stato della posizione
    in_position = False
    
    for i in range(len(entry)):
        # Se abbiamo un segnale di entrata mentre siamo già in posizione, lo rimuoviamo
        if clean_entry[i] and in_position:
            clean_entry[i] = False
        
        # Se abbiamo un segnale di uscita mentre non siamo in posizione, lo rimuoviamo
        if clean_exit[i] and not in_position:
            clean_exit[i] = False
        
        # Aggiorniamo lo stato della posizione
        if clean_entry[i]:
            in_position = True
        if clean_exit[i]:
            in_position = False
            
    return clean_entry, clean_exit
def get_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bullish, bearish = get_pattern(close, open_, high, low)
    rsi      = talib.RSI(close, timeperiod=14)
    upper, _ , lower = talib.BBANDS(close, timeperiod=14, nbdevup=bb_std, nbdevdn=bb_std)
    bb_width = (upper - lower)

    entry_long = (rsi < rsi_entry) & (close < lower) & bullish & (bb_width < bb_width_threshold)
    entry_short = (rsi > rsi_exit)  & (close > upper) & bearish & (bb_width < bb_width_threshold)
    exit_long = (rsi > rsi_exit)  & (close > upper) & bearish & (bb_width < bb_width_threshold)
    exit_short = (rsi < rsi_entry) & (close < lower) & bullish & (bb_width < bb_width_threshold)
    
    # Pulisco i segnali di entrata e uscita
    entry_long, exit_long = clean_signal(entry_long, exit_long)
    entry_short, exit_short = clean_signal(entry_short, exit_short)
    return entry_long, exit_long, entry_short, exit_short
def get_ohlcv(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    df = pd.read_csv(file_path, header=None, names=column_names)
    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    return open_, high, low, close
def get_exits_limit(close: np.ndarray, atr: np.ndarray, entry: np.ndarray, exit: np.ndarray, isLong: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola gli stop loss e take profit per ogni segnale di entrata e genera segnali di uscita quando vengono raggiunti.
    
    Parameters:
    -----------
    close: np.ndarray - Array dei prezzi di chiusura
    atr: np.ndarray - Average True Range per calcolare i livelli di TP/SL
    entry: np.ndarray - Array booleano dei segnali di entrata
    exit: np.ndarray - Array booleano dei segnali di uscita esistenti
    isLong: bool - True per posizioni long, False per posizioni short
    
    Returns:
    --------
    tuple[np.ndarray, np.ndarray] - (exit_array modificato, prezzi_limite)
    """
    # Copio l'array di uscita per non modificare l'originale
    exit_array = exit.copy()
    # Creo un array per i prezzi limite (TP o SL raggiunti)
    price_limit = np.full(len(close), np.nan)
    
    # Variabili per tenere traccia dello stato
    in_position = False
    entry_price = 0.0
    entry_index = 0
    tp_level = 0.0
    sl_level = 0.0
    
    for i in range(len(close)):
        # Se c'è un segnale di entrata e non siamo già in posizione
        if entry[i] and not in_position:
            in_position = True
            entry_price = close[i]
            entry_index = i
            
            # Calcola TP e SL in base alla direzione
            if isLong:
                tp_level = entry_price + (tp_mult * atr[i])
                sl_level = entry_price - (sl_mult * atr[i])
            else:
                tp_level = entry_price - (tp_mult * atr[i])
                sl_level = entry_price + (sl_mult * atr[i])
        
        # Se siamo in posizione, verifichiamo se è necessario uscire
        if in_position:
            # Verifica se il prezzo ha raggiunto TP o SL
            if isLong:
                hit_tp = close[i] >= tp_level
                hit_sl = close[i] <= sl_level
            else:
                hit_tp = close[i] <= tp_level
                hit_sl = close[i] >= sl_level
            
            # Se viene raggiunto il take profit
            if hit_tp:
                exit_array[i+1] = True
                price_limit[i+1] = tp_level
                in_position = False
            
            # Se viene raggiunto lo stop loss
            elif hit_sl:
                exit_array[i+1] = True
                price_limit[i+1] = sl_level
                in_position = False
            
            # Se la posizione è scaduta (expiration)
            elif i - entry_index >= expiration:
                exit_array[i+1] = True
                price_limit[i+1] = close[i]
                in_position = False
            
            # Se era già presente un segnale di uscita
            elif exit[i]:
                price_limit[i+1] = close[i]
                in_position = False
    
    return exit_array, price_limit
def get_limits(price_limit_long: np.ndarray, price_limit_short: np.ndarray) -> np.ndarray:
    # Creo un vettore price_limit combinando i valori di price_limit_long e price_limit_short
    price_limit = np.full(len(price_limit_long), np.nan)

    # Aggiungo i valori di price_limit_long dove non sono NaN
    mask_long = ~np.isnan(price_limit_long)
    price_limit[mask_long] = price_limit_long[mask_long]

    # Aggiungo i valori di price_limit_short dove non sono NaN
    mask_short = ~np.isnan(price_limit_short)
    price_limit[mask_short] = price_limit_short[mask_short]
    return price_limit
open_, high, low, close = get_ohlcv(DATA_FILE)
atr = talib.ATR(high, low, close, timeperiod=14)
entry_long, exit_long, entry_short, exit_short = get_signal()
exit_long, price_limit_long = get_exits_limit(close, atr, entry_long, exit_long, True)
exit_short, price_limit_short = get_exits_limit(close, atr, entry_short, exit_short, False)
price_limits = get_limits(price_limit_long, price_limit_short)

portfolio = vbt.Portfolio.from_signals(
    close = close,
    open = open_,
    high = high,
    low = low,
    freq = '1min',
    entries = entry_long,
    exits = exit_long,
    short_entries = entry_short,
    short_exits = exit_short,
    size = exposure_pct,
    size_type = 'percent',
    limit_order_price = price_limits,
    init_cash = init_cash,
    leverage = 30,
    fees = 0.0000022,
    limit_expiry = expiration,
)   

print(portfolio.stats())