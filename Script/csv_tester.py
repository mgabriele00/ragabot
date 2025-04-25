import pandas as pd
import talib
import numpy as np

# --- Parametri ---
INPUT_CSV_FILE = 'rates_data.csv'  # Il file CSV generato dal bot
OUTPUT_CSV_FILE = 'rates_with_rsi.csv' # Il nuovo file CSV con l'RSI
RSI_PERIOD = 14 # Periodo standard per l'RSI

# --- Lettura del CSV ---
try:
    # Legge il CSV usando pandas. Assicurati che la prima riga sia l'header.
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"✅ Letto {len(df)} righe da {INPUT_CSV_FILE}")

    # Verifica che la colonna 'close' esista
    if 'close' not in df.columns:
        raise ValueError("La colonna 'close' non è presente nel file CSV.")

    # Assicurati che la colonna 'close' sia numerica (float)
    # Se ci sono valori non numerici, verranno convertiti in NaN e poi gestiti da talib
    close_prices = pd.to_numeric(df['close'], errors='coerce')

    # --- Calcolo RSI ---
    # Calcola l'RSI usando TA-Lib.
    # Nota: talib.RSI restituisce un array numpy. I primi 'RSI_PERIOD - 1' valori saranno NaN
    # perché l'RSI ha bisogno di un certo numero di periodi precedenti per essere calcolato.
    rsi_values = talib.RSI(close_prices.values, timeperiod=RSI_PERIOD)
    print(f"✅ Calcolato RSI con periodo {RSI_PERIOD}")

    # --- Aggiunta colonna RSI al DataFrame ---
    # Aggiunge la serie di valori RSI calcolati come nuova colonna nel DataFrame.
    # Pandas allineerà automaticamente i valori basandosi sull'indice.
    df['rsi'] = rsi_values

    # --- Salvataggio del nuovo CSV ---
    # Salva il DataFrame modificato (con la nuova colonna 'rsi') in un nuovo file CSV.
    # index=False evita che pandas scriva l'indice del DataFrame come colonna nel CSV.
    df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.5f') # Mantieni la formattazione per i float
    print(f"✅ Dati con RSI salvati in {OUTPUT_CSV_FILE}")

except FileNotFoundError:
    print(f"❌ Errore: File non trovato: {INPUT_CSV_FILE}")
except ValueError as ve:
    print(f"❌ Errore nei dati: {ve}")
except Exception as e:
    print(f"❌ Errore generico durante l'elaborazione: {e}")
