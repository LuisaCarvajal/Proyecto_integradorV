import pandas as pd
import numpy as np
import yfinance as yf

def enrich_data(df):
    print(df.columns)  # para verificar cómo se llama la columna real

    # Encuentra la columna que contiene el precio de cierre (ej. close_005930.KS)
    for col in df.columns:
        if col.startswith("close_"):
            df = df.rename(columns={col: "close_samsung"})
            break

    if "close_samsung" not in df.columns:
        raise ValueError("No se encontró una columna que empiece por 'close_'")

    df["daily_return"] = df["close_samsung"].pct_change()
    df["rolling_mean_20"] = df["close_samsung"].rolling(window=20).mean()
    df["rolling_std_20"] = df["close_samsung"].rolling(window=20).std()
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()
    df["target"] = df["daily_return"].shift(-1) > 0

    return df


def add_macro_indicator(df, ticker_macro="^KS11"):
    # Validar que exista la columna 'date'
    if "date" not in df.columns:
        raise ValueError("La columna 'date' no está en el DataFrame.")
    
    # Descargar datos macroeconómicos desde Yahoo Finance
    macro = yf.download(ticker_macro, start=df["date"].min(), end=df["date"].max(), progress=False)
    
    # Seleccionar solo la columna Close y resetear índice
    macro = macro[['Close']].reset_index()
    
    # Renombrar columnas manualmente para evitar problemas con MultiIndex y .str accessor
    macro.columns = ['date', 'kospi_close']

    # Asegurar que 'date' sea columna normal en df, no índice
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Unir macroeconómico con el dataframe principal por 'date'
    df = pd.merge(df, macro[['date', 'kospi_close']], on="date", how="left")
    
    # Calcular retorno del índice macroeconómico
    df["kospi_return"] = df["kospi_close"].pct_change()
    
    return df
