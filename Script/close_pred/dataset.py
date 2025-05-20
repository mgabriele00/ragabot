import numpy as np
import polars as pl
import talib

# Carica dati CSV con Polars
def load_data(csv_path):
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        (pl.col('Date') + ' ' + pl.col('Time')).alias('datetime')
    ).drop(['Date', 'Time'])
    df = df.with_columns(pl.col('datetime').str.to_datetime('%Y.%m.%d %H:%M'))
    df = df.set_sorted('datetime')
    return df

# Calcolo feature tecniche + variabili storiche
def calculate_features(df, history_bars=5):
    close = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    open_ = df['Open'].to_numpy()

    features = {
        'ATR': talib.ATR(high, low, close, timeperiod=14),
        'RSI': talib.RSI(close, timeperiod=14),
    }

    upper, mid, lower = talib.BBANDS(close, timeperiod=14, nbdevup=2, nbdevdn=2)
    features.update({
        'BB_upper': upper,
        'BB_mid': mid,
        'BB_lower': lower,
        'SMA_10': talib.SMA(close, 10),
        'SMA_20': talib.SMA(close, 20),
        'EMA_10': talib.EMA(close, 10),
        'EMA_20': talib.EMA(close, 20),
    })

    macd, macdsignal, macdhist = talib.MACD(close, 12, 26, 9)
    features.update({
        'MACD': macd,
        'MACD_signal': macdsignal,
        'MACD_hist': macdhist,
    })

    # Supertrend personalizzato
    def supertrend(high, low, close, period=10, multiplier=3):
        atr = talib.ATR(high, low, close, period)
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)

        trend = np.zeros(len(close))
        direction = 1

        for i in range(1, len(close)):
            if close[i] > upperband[i - 1]:
                direction = 1
            elif close[i] < lowerband[i - 1]:
                direction = -1
            trend[i] = upperband[i] if direction == -1 else lowerband[i]
        return trend

    features['Supertrend'] = supertrend(high, low, close)

    # Pattern candlestick TA-Lib
    patterns = talib.get_function_groups()["Pattern Recognition"]
    for pattern in patterns:
        func = getattr(talib, pattern)
        features[pattern] = np.sign(func(open_, high, low, close))

    features_df = pl.DataFrame(features)

    # Shift delle feature e dei prezzi grezzi
    all_shifts = []
    for i in range(1, history_bars + 1):
        shifted_features = features_df.select([pl.col(c).shift(i) for c in features_df.columns])
        shifted_features = shifted_features.rename({col: f"{col}_t{i}" for col in shifted_features.columns})
        all_shifts.append(shifted_features)

    # Shift anche di Close, High, Low
    for price_col in ['Close', 'High', 'Low']:
        for i in range(1, history_bars + 1):
            df = df.with_columns(pl.col(price_col).shift(i).alias(f"{price_col}_t{i}"))

    all_shifted = pl.concat(all_shifts, how="horizontal")
    return df.hstack(features_df).hstack(all_shifted)

# Calcola livelli teorici TP e SL
def calculate_tp_sl(df, features, tp_mult=2, sl_mult=1):
    close = df['Close'].to_numpy()
    atr = features['ATR'].to_numpy()

    tp = close + tp_mult * atr
    sl = close - sl_mult * atr

    features = features.with_columns([
        pl.Series('TP_theoretical', tp),
        pl.Series('SL_theoretical', sl)
    ])
    return features

# Main workflow
def create_feature_dataset(csv_path):
    df = load_data(csv_path)
    df_with_features = calculate_features(df)
    df_with_tp_sl = calculate_tp_sl(df_with_features, df_with_features)

    final_df = df_with_tp_sl.drop_nulls()
    return final_df

# Esecuzione
if __name__ == "__main__":
    csv_path = '../close_pred/data/EURUSD_M1_2013_2024.csv'
    feature_dataset = create_feature_dataset(csv_path)
    feature_dataset.write_parquet('feature_dataset.parquet')
    print("Dataset con feature tecniche, storiche e livelli TP/SL salvato correttamente.")
