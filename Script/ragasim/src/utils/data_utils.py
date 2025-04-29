import pandas as pd
import numpy as np

# funzione per leggere un csv senza header e restituire array numpy di date, time, open, high, low, close
def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a CSV file without header and return numpy arrays for date, time, open, high, low, and close prices.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        tuple: A tuple containing numpy arrays for date, time, open, high, low, and close prices.
    """
    # il csv non ha intestazione, impostiamo i nomi delle colonne
    data = pd.read_csv(
        file_path,
        header=None,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )
    date  = data['Date'].to_numpy()
    time  = data['Time'].to_numpy()
    open_ = data['Open'].to_numpy()
    high  = data['High'].to_numpy()
    low   = data['Low'].to_numpy()
    close = data['Close'].to_numpy()
    
    return date, time, open_, high, low, close