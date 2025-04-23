# ─── Logica di generazione segnali (adattata dallo script originale) ──────────
from typing import List, Tuple
import numpy as np
import talib
from datetime import datetime # Aggiunto per coerenza logging
from ..models.signal_models import BarData, CloseOrder, OpenOrder, SignalResponse, SignalRequest # Aggiunto OrderResource, SignalRequestDto

def generate_signals_api(rates_list: List[BarData], params: dict) -> Tuple[bool, bool]:
    """
    Genera segnali basandosi sulla lista di TickData.
    Restituisce l'ultimo segnale di entry e exit.
    """
    # Converti la lista di Pydantic models in un formato utilizzabile da talib
    # Estrai solo i campi necessari (ohlc) in array numpy
    opens = np.array([tick.open for tick in rates_list], dtype=float)
    highs = np.array([tick.high for tick in rates_list], dtype=float)
    lows = np.array([tick.low for tick in rates_list], dtype=float)
    closes = np.array([tick.close for tick in rates_list], dtype=float)

    if len(closes) < 15: # Controllo minimo per RSI(14) e BBANDS(14)
        print("   [Warning] Dati insufficienti per calcolare indicatori.")
        return False, False # Nessun segnale se non ci sono abbastanza dati

    # RSI
    rsi = talib.RSI(closes, timeperiod=14)

    # Pattern recognition (come nello script originale)
    bullish = np.zeros_like(closes, dtype=bool)
    bearish = np.zeros_like(closes, dtype=bool)
    try:
        # ... (codice pattern recognition invariato) ...
        for pat in talib.get_function_groups()["Pattern Recognition"]:
            if hasattr(talib, pat):
                vals = getattr(talib, pat)(opens, highs, lows, closes)
                if len(vals) == len(bullish):
                    bullish |= (vals > 0)
                    bearish |= (vals < 0)
                else:
                     pass
    except Exception as e:
        print(f"   [Error] Errore nel calcolo pattern recognition: {e}")


    # Bollinger Bands
    upper, _, lower = talib.BBANDS(
        closes,
        timeperiod=14,
        nbdevup=params["bb_std"],
        nbdevdn=params["bb_std"]
    )

    # Vettorializza segnali
    valid_indices = np.where(
        ~np.isnan(rsi) & ~np.isnan(upper) & ~np.isnan(lower)
    )[0]

    if len(valid_indices) == 0:
         print("   [Warning] Indicatori non ancora validi.")
         return False, False

    last_valid_idx = valid_indices[-1]

    # Calcola segnali sull'ultimo dato valido
    last_entry = (rsi[last_valid_idx] < params["rsi_entry"]) and \
                 (closes[last_valid_idx] < lower[last_valid_idx]) and \
                 bullish[last_valid_idx]
    last_exit = (rsi[last_valid_idx] > params["rsi_exit"]) and \
                (closes[last_valid_idx] > upper[last_valid_idx]) and \
                bearish[last_valid_idx]

    print(f"   Latest valid tick signals (@ index {last_valid_idx}): entry={last_entry}, exit={last_exit}")
    return last_entry, last_exit


def determine_order_action(request: SignalRequest, params: dict) -> SignalResponse:
    """
    Determina l'azione dell'ordine (entry, exit, hold) basandosi sui segnali,
    il conteggio delle posizioni e calcola i parametri dell'ordine se necessario.
    """
    # 1. Genera segnali grezzi
    # Assicurati che request.ticks non sia vuoto prima di chiamare questa funzione (fatto nel router)
    last_entry, last_exit = generate_signals_api(request.bars, params)

   # Creazione di un'istanza OpenOrder moccata
    mock_open_order = OpenOrder(
        symbol="EURUSD",
        type="buy",
        volume=0.1,
        stop_loss=1.01000,
        take_profit=1.19500,
        comment="Entry signal based on RSI/BB"
    )

    # Creazione di un'istanza CloseOrder moccata
    mock_close_order = CloseOrder(
        symbol="EURUSD",
        ticket=146347654, # Ticket della posizione da chiudere
        comment="Exit signal based on RSI/BB"
    )
    
    mock_close_order1 = CloseOrder(
        symbol="EURUSD",
        ticket=146347655, # Ticket della posizione da chiudere
        comment="Exit signal based on RSI/BB"
    )
     
    mock_close_order2 = CloseOrder(
        symbol="EURUSD",
        ticket=146352290, # Ticket della posizione da chiudere
        comment="Exit signal based on RSI/BB"
    )
     
    mock_close_order3 = CloseOrder(
        symbol="EURUSD",
        ticket=146352294, # Ticket della posizione da chiudere
        comment="Exit signal based on RSI/BB")

    # Creazione di un'istanza SignalResponse moccata
    mock_signal_response = SignalResponse(
        orders_to_open=[mock_open_order, mock_open_order],
        orders_to_close=[mock_close_order, mock_close_order1, mock_close_order2, mock_close_order3],
    )
    
    return mock_signal_response