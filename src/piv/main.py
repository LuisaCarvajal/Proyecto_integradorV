from collector import DataCollector
from enricher import enrich_data, add_macro_indicator
from modeller import entrenar, predecir, evaluar
import pandas as pd
#from modeller import Modeller

if __name__ == "__main__":
    # 1. Recolectar datos
    collector = DataCollector()
    df = collector.fetch_data()

    if df.empty:
        print("No se obtuvieron datos.")
        exit()

    # 2. Enriquecer con variables financieras
    df = enrich_data(df)  # Usa el sufijo en los nombres de columna
    
    # 3. Agregar indicador macroeconómico
    df = add_macro_indicator(df, ticker_macro="^KS11")  # Cambia si deseas otro índice

    # 4. Guardar actualizaciones (sin sobrescribir encabezados)
    collector.update_csv(df)
    collector.update_sqlite(df)

    # 5. Preparar para entrenamiento
    feature_cols = [
        "daily_return", "rolling_mean_20", "rolling_std_20",
        "volatility_20", "kospi_return"
    ]
    target_col = "target"

    if not all(col in df.columns for col in feature_cols + [target_col]):
        raise ValueError("Faltan columnas necesarias para entrenamiento.")

    X = df[feature_cols]
    y = df[target_col]

    # 6. Dividir datos (ejemplo simple)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 7. Entrenar y evaluar
    entrenar(X_train, y_train)
    y_pred = predecir(X_test)
    evaluar(y_test, y_pred)
