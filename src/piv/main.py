from collector import DataCollector
from enricher import enrich_data, add_macro_indicator
from modeller import entrenar_y_evaluar

if __name__ == "__main__":
    #Recolectar datos
    collector = DataCollector()
    df = collector.fetch_data()

    if df.empty:
        print("No se obtuvieron datos.")
        exit()

    #Enriquecer
    df = enrich_data(df)
    df = add_macro_indicator(df)
    
    #Guardar
    collector.update_csv(df)
    collector.update_sqlite(df)

    #Entrenar y evaluar en modeller.py
    entrenar_y_evaluar(df)
