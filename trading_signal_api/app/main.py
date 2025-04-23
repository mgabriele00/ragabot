from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import json
import time # Per timestamp nel log

from .routers import trading_router

app = FastAPI(title="Trading Signal API")

# Middleware per loggare il corpo di tutte le richieste
class LogRequestBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_body_bytes = await request.body()
        # È importante leggere il corpo qui. Per permettere all'endpoint
        # di leggerlo di nuovo, dobbiamo "ricostruire" lo stream.
        # FastAPI/Starlette lo gestiscono internamente se usiamo request.body()
        # ma per essere sicuri, lo rimettiamo nello scope della richiesta.
        # Nota: Questo è gestito automaticamente da Starlette > 0.20.0
        # request.scope['body_stream_consumed'] = False # Non più necessario di solito

        # Logga il corpo
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- [{timestamp}] Richiesta Ricevuta: {request.method} {request.url.path} ---")
        try:
            # Prova a decodificare e stampare come JSON formattato
            req_body_json = json.loads(req_body_bytes.decode() if req_body_bytes else '{}')
            print("--- Corpo Richiesta (JSON) ---")
            print(json.dumps(req_body_json, indent=2))
        except json.JSONDecodeError:
            # Se non è JSON, stampa come stringa (o bytes se la decodifica fallisce)
            print("--- Corpo Richiesta (Raw) ---")
            try:
                print(req_body_bytes.decode())
            except UnicodeDecodeError:
                print(req_body_bytes)
        print("--------------------------------------------------------------------")

        # Procedi con la gestione della richiesta
        response = await call_next(request)
        return response

# Aggiungi il middleware all'applicazione
app.add_middleware(LogRequestBodyMiddleware)

app.include_router(trading_router.router, prefix="/v1/trading", tags=["signals"])

# Nota: Se hai bisogno di eseguire codice all'avvio (come caricare modelli ML),
# puoi usare gli eventi startup/shutdown di FastAPI.
# @app.on_event("startup")
# async def startup_event():
#     print("Avvio API...")
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     print("Spegnimento API...")