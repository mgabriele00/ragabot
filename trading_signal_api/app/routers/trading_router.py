# ─── Router FastAPI ─────────────────────────────────────────────────────
from datetime import datetime
from fastapi import APIRouter, HTTPException
from ..config.settings import PARAMS
# Importa la nuova funzione di servizio, rimuovi generate_signals_api
from ..services.trading_service import determine_order_action
from ..models.signal_models import SignalResponse, SignalRequest

router = APIRouter()

@router.post("/signal", response_model=SignalResponse) # Endpoint rimane /signal
async def get_trading_signal(request: SignalRequest) -> SignalResponse:
    """
    Riceve dati tick, saldo, simbolo e conteggio posizioni.
    Determina e restituisce l'azione di trading appropriata (entry, exit, hold)
    con parametri calcolati se applicabile, delegando la logica al servizio.
    """
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] === Richiesta /signal ricevuta ===")
    # Logga anche il nuovo parametro position_count
    print(f"   Symbol: {request.symbol}, Balance: {request.balance}, Ticks ricevuti: {len(request.bars)}, Posizioni Aperte: {request.open_positions}")

    # La validazione di base della richiesta rimane nel router
    if not request.bars:
         raise HTTPException(status_code=400, detail="La lista 'ticks' non può essere vuota.")

    # Chiama la funzione di servizio per gestire tutta la logica e restituire il risultato
    order_result = determine_order_action(request, PARAMS)

    # Restituisce direttamente l'oggetto OrderResource creato dal servizio
    return order_result