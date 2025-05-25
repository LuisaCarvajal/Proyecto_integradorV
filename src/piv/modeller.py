# src/modeller.py
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

MODEL_PATH = "src/piv/static/models/model.pkl"

def entrenar(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo entrenado y guardado en {MODEL_PATH}")

def predecir(X_test):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Entrénalo primero.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    return predictions

def evaluar(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
   #mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    #print(f"MAPE: {mape:.2f}%")

    return rmse, mae #, mape

def entrenar_y_evaluar(df):
    feature_cols = [
        "daily_return", "rolling_mean_20", "rolling_std_20",
        "volatility_20", "kospi_return"
    ]
    target_col = "target"

    df = df.dropna(subset=feature_cols + [target_col])

    if not all(col in df.columns for col in feature_cols + [target_col]):
        raise ValueError("Faltan columnas necesarias para entrenamiento.")

    X = df[feature_cols]
    y = df[target_col]

    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    entrenar(X_train, y_train)
    y_pred = predecir(X_test)
    return evaluar(y_test, y_pred)
