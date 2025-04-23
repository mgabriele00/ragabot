# ─── Modelli Pydantic per i dati di input e output ────────────────────────────
from typing import List, Optional
from pydantic import BaseModel, Field

class BarData(BaseModel):
    """Modello per un singolo tick/rate, simile alla struttura MT5."""
    timestamp: int # Timestamp in secondi since epoch
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Aggiungi altri campi se necessari (es. tick_volume, spread, real_volume)
    # ma generate_signals usa solo ohlc
    
class OpenPosition(BaseModel):
    """Modello per una posizione aperta."""
    symbol: str
    ticket: int
    type: str # "buy" o "sell"
    volume: float
    open_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
class OpenOrder(BaseModel):    
    """Modello per aprire un ordine."""
    symbol: str
    type: str # "buy" o "sell"
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None # Commento opzionale per l'ordine
    
class CloseOrder(BaseModel):
    """Modello per chiudere un ordine."""
    symbol: str
    ticket: int
    comment: Optional[str] = None # Commento opzionale per la chiusura    
    
class SignalRequest(BaseModel): # Rinominato da SignalRequest
    """Modello per la richiesta all'API."""
    symbol: str # Aggiunto simbolo
    bars: List[BarData] = Field(..., min_items=1) # Richiede almeno 15 tick per calcolare RSI(14)
    balance: float
    open_positions: List[OpenPosition] = [] # Lista di posizioni aperte    
    
class SignalResponse(BaseModel):
    """Modello per la risposta dell'API."""
    orders_to_open: List[OpenOrder] = [] # Lista di ordini da aprire
    orders_to_close: List[CloseOrder] = [] # Lista di ordini da chiudere